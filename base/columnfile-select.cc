#include "base/columnfile.h"

namespace ev {

ColumnFileSelect::ColumnFileSelect(ColumnFileReader input)
    : input_(std::move(input)) {}

void ColumnFileSelect::AddSelection(uint32_t field) {
  selection_.emplace(field);
}

void ColumnFileSelect::AddFilter(
    uint32_t field, std::function<bool(const StringRefOrNull&)> filter) {
  filters_.emplace_back(field, std::move(filter));
}

void ColumnFileSelect::StartScan() {
  if (filters_.empty()) {
    input_.SetColumnFilter(selection_.begin(), selection_.end());
    return;
  }

  row_buffer_.clear();

  // Sort filters by column index.
  std::stable_sort(
      filters_.begin(), filters_.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

  unfiltered_selection_ = selection_;
  for (const auto& filter : filters_) unfiltered_selection_.erase(filter.first);
}

void ColumnFileSelect::ReadChunk(ev::concurrency::RegionPool& region_pool) {
  for (;;) {
    row_buffer_.clear();
    chunk_offset_ = 0;

    // Used to hold temporary copies of strings.
    region_ = region_pool.GetRegion();

    size_t filter_idx = 0;

    // Iterate over all filters.
    do {
      auto field = filters_[filter_idx].first;

      input_.SetColumnFilter({field});
      if (filter_idx == 0) {
        if (input_.End()) return;
      } else {
        input_.SeekToStartOfSegment();
      }

      auto filter_selected = selection_.count(field);

      size_t filter_range_end = filter_idx + 1;

      while (filter_range_end != filters_.size() &&
             filters_[filter_range_end].first == field)
        ++filter_range_end;

      auto in = row_buffer_.begin();
      auto out = row_buffer_.begin();

      // Iterate over all values in the current segment for the current column.
      for (uint32_t row_idx = 0; !input_.EndOfSegment(); ++row_idx) {
        auto row = input_.GetRow();

        if (filter_idx > 0) {
          // Is row already filtered?
          if (row_idx < in->index) continue;

          KJ_ASSERT(in->index == row_idx);
        }

        ev::StringRefOrNull value = nullptr;
        if (row.size() == 1) {
          KJ_ASSERT(row[0].first == field, row[0].first, field);
          value = row[0].second;
        }

        bool match = true;

        for (size_t i = filter_idx; i != filter_range_end; ++i) {
          if (!filters_[i].second(value)) {
            match = false;
            break;
          }
        }

        if (match) {
          if (filter_idx == 0) {
            RowCache row_cache;
            row_cache.index = row_idx;

            if (filter_selected) {
              if (value.IsNull())
                row_cache.data.emplace_back(field, nullptr);
              else
                row_cache.data.emplace_back(field,
                                            value.StringRef().dup(region_));
            }

            row_buffer_.emplace_back(std::move(row_cache));
          } else {
            if (out != in) *out = std::move(*in);

            if (filter_selected) {
              if (value.IsNull())
                out->data.emplace_back(field, nullptr);
              else
                out->data.emplace_back(field, value.StringRef().dup(region_));
            }

            ++out;
            ++in;
          }
        }
      }

      if (filter_idx > 0) row_buffer_.erase(out, row_buffer_.end());

      filter_idx = filter_range_end;
    } while (!row_buffer_.empty() && filter_idx < filters_.size());

    // Now rows passing filter in current chunk, read next one.
    if (row_buffer_.empty()) continue;

    if (!unfiltered_selection_.empty()) {
      input_.SetColumnFilter(unfiltered_selection_.begin(),
                             unfiltered_selection_.end());
      input_.SeekToStartOfSegment();

      auto sr = row_buffer_.begin();

      for (uint32_t row_idx = 0; !input_.EndOfSegment(); ++row_idx) {
        if (sr == row_buffer_.end()) break;

        auto row = input_.GetRow();

        if (row_idx < sr->index) continue;

        KJ_REQUIRE(row_idx == sr->index, row_idx, sr->index);

        for (const auto& d : row) {
          const auto& value = d.second;
          if (value.IsNull())
            sr->data.emplace_back(d.first, nullptr);
          else
            sr->data.emplace_back(d.first, d.second.StringRef().dup(region_));
        }

        ++sr;
      }

      while (!input_.EndOfSegment()) input_.GetRow();
    }

    break;
  }
}

kj::Maybe<const std::vector<std::pair<uint32_t, StringRefOrNull>>&>
ColumnFileSelect::Iterate(ev::concurrency::RegionPool& region_pool) {
  if (filters_.empty()) {
    if (input_.End()) return nullptr;

    return input_.GetRow();
  }

  if (chunk_offset_ == row_buffer_.size()) {
    ReadChunk(region_pool);
    if (chunk_offset_ == row_buffer_.size()) return nullptr;
  }

  auto& row = row_buffer_[chunk_offset_++];
  std::sort(
      row.data.begin(), row.data.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

  return row.data;
}

void ColumnFileSelect::Execute(
    ev::concurrency::RegionPool& region_pool,
    std::function<void(
        const std::vector<std::pair<uint32_t, StringRefOrNull>>&)> callback) {
  StartScan();

  for (;;) {
    KJ_IF_MAYBE(row, Iterate(region_pool)) { callback(*row); }
    else {
      break;
    }
  }
}

}  // namespace ev
