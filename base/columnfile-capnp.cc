#include "base/columnfile-capnp.h"

namespace ev {

namespace {

// Returns the number of columns required to encode the given type.
//
// The `depth` parameter is used to protect against infinite recursion, and
// generally against extremely deep type hierarchies.
uint32_t FieldCount(const capnp::Type& type, size_t depth = 0) {
  if (type.isVoid()) return 0;

  // TODO(mortehu): Cache result

  KJ_REQUIRE(depth < 256, depth);

  uint32_t result = 0;

  if (type.isStruct()) {
    for (const auto field : type.asStruct().getFields())
      result += FieldCount(field.getType(), depth + 1);

    return result;
  }

  if (type.isList()) {
    const auto element_type = type.asList().getElementType();
    if (element_type.isList() || element_type.isStruct())
      return 1 + FieldCount(element_type, depth + 1);

    KJ_REQUIRE(!element_type.isAnyPointer());
    KJ_REQUIRE(!element_type.isInterface());

    return 2;
  }

  KJ_REQUIRE(!type.isAnyPointer());
  KJ_REQUIRE(!type.isInterface());

  return 1;
}

}  // namespace

void WriteMessageToColumnFile(ColumnFileWriter& output,
                              capnp::DynamicValue::Reader message) {
  struct Context {
    Context(capnp::DynamicValue::Reader value, uint32_t column)
        : value(std::move(value)), column(column) {}

    Context(Context&&) = default;

    Context& operator=(Context&&) = default;

    KJ_DISALLOW_COPY(Context);

    capnp::DynamicValue::Reader value;
    uint32_t column;
  };

  std::deque<Context> queue;
  queue.emplace_back(message, 0);

  while (!queue.empty()) {
    Context ctx = std::move(queue.front());
    queue.pop_front();

    auto column = ctx.column;

    switch (ctx.value.getType()) {
      case capnp::DynamicValue::LIST: {
        const auto list = ctx.value.as<capnp::DynamicList>();
        auto element_type = list.getSchema().getElementType();
        uint64_t list_size = list.size();
        output.Put(column++,
                   ev::StringRef(reinterpret_cast<const char*>(&list_size),
                                 sizeof(list_size)));
        if (element_type.isList() || element_type.isStruct()) {
          for (size_t i = 0; i < list.size(); ++i)
            queue.emplace_back(list[i], column);
          column += FieldCount(element_type);
        } else {
          KJ_REQUIRE(!element_type.isAnyPointer());
          KJ_REQUIRE(!element_type.isInterface());
          KJ_FAIL_REQUIRE("missing implementation");
        }
      } break;

      case capnp::DynamicValue::STRUCT: {
        auto struct_value = ctx.value.as<capnp::DynamicStruct>();
        // TODO(mortehu): Also deal with `getUnionFields`

        auto emit_field = [&struct_value, &column, &output,
                           &queue](capnp::StructSchema::Field field) {
          KJ_CONTEXT(field.getProto().getName());

          auto field_type = field.getType();

          if (field_type.isStruct()) {
            const auto nested_field_count = FieldCount(field_type);
            if (nested_field_count > 0) {
              queue.emplace_back(struct_value.get(field), column);
              column += nested_field_count;
            }
          } else if (field_type.isList()) {
            const auto nested_field_count = FieldCount(field_type);
            queue.emplace_back(struct_value.get(field), column);
            column += nested_field_count;
          } else if (field_type.isVoid()) {
            // Do nothing.
          } else if (!struct_value.has(field)) {
            output.PutNull(column++);
          } else if (field_type.isEnum()) {
            auto value =
                struct_value.get(field).as<capnp::DynamicEnum>().getRaw();
            output.Put(column++,
                       ev::StringRef(reinterpret_cast<const char*>(&value),
                                     sizeof(value)));
          } else if (field_type.isInt32()) {
            auto value = struct_value.get(field).as<int32_t>();
            output.Put(column++,
                       ev::StringRef(reinterpret_cast<const char*>(&value),
                                     sizeof(value)));
          } else if (field_type.isUInt32()) {
            auto value = struct_value.get(field).as<uint32_t>();
            output.Put(column++,
                       ev::StringRef(reinterpret_cast<const char*>(&value),
                                     sizeof(value)));
          } else if (field_type.isUInt64()) {
            auto value = struct_value.get(field).as<uint64_t>();
            output.Put(column++,
                       ev::StringRef(reinterpret_cast<const char*>(&value),
                                     sizeof(value)));
          } else if (field_type.isText() || field_type.isData()) {
            auto value = struct_value.get(field).as<capnp::Text>();
            output.Put(column++, ev::StringRef(value));
          } else {
            KJ_FAIL_REQUIRE("Unhandled field type");
          }
        };

        for (auto field : struct_value.getSchema().getNonUnionFields())
          emit_field(field);
      } break;

      default:
        KJ_FAIL_REQUIRE("Unhandled type", ctx.value.getType());
    }
  }
}

void ReadMessageFromColumnFile(ColumnFileReader& input,
                               capnp::DynamicValue::Builder output) {
  struct Context {
    Context(capnp::DynamicValue::Builder value, uint32_t column)
        : value(std::move(value)), column(column) {}

    Context(Context&&) = default;

    Context& operator=(Context&&) = default;

    KJ_DISALLOW_COPY(Context);

    capnp::DynamicValue::Builder value;
    uint32_t column;
  };

  std::deque<Context> queue;
  queue.emplace_back(output, 0);

  while (!queue.empty()) {
    Context ctx = std::move(queue.front());
    queue.pop_front();

    auto column = ctx.column;
    KJ_CONTEXT(column);

    switch (ctx.value.getType()) {
      case capnp::DynamicValue::VOID:
        break;

      case capnp::DynamicValue::LIST: {
        auto list = ctx.value.as<capnp::DynamicList>();
        auto element_type = list.getSchema().getElementType();
        // List must be already initialized by caller.
        if (element_type.isList() || element_type.isStruct()) {
          ++column;
          for (size_t i = 0; i < list.size(); ++i)
            queue.emplace_back(list[i], column);
          column += FieldCount(element_type);
        } else {
          KJ_REQUIRE(!element_type.isAnyPointer());
          KJ_REQUIRE(!element_type.isInterface());
          KJ_FAIL_REQUIRE("missing implementation");
        }
      } break;

      case capnp::DynamicValue::STRUCT: {
        auto struct_value = ctx.value.as<capnp::DynamicStruct>();
        // TODO(mortehu): Deal with getUnionFields
        for (auto field : struct_value.getSchema().getNonUnionFields()) {
          KJ_CONTEXT(field.getProto().getName());
          auto field_type = field.getType();

          if (field_type.isStruct()) {
            const auto nested_field_count = FieldCount(field_type);
            if (nested_field_count > 0) {
              queue.emplace_back(struct_value.init(field), column);
              column += nested_field_count;
            }
          } else if (field_type.isList()) {
            auto data = input.Get(column);
            KJ_REQUIRE_NONNULL(data);
            KJ_REQUIRE(data->size() == sizeof(uint64_t), data->size());
            const auto count = *reinterpret_cast<const uint64_t*>(data->data());

            if (count)
              queue.emplace_back(struct_value.init(field, count), column);

            column += FieldCount(field_type);
          } else {
            auto data = input.Get(column++);

            // Skip null values.
            if (!data) continue;

            if (field_type.isBool()) {
              KJ_REQUIRE(data->size() == 1, data->size());
              struct_value.set(field, (*data)[0] != 0);
            } else if (field_type.isUInt32()) {
              KJ_REQUIRE(data->size() == 4, data->size());
              struct_value.set(field, *(const uint32_t*)data->data());
            } else if (field_type.isText() || field_type.isData()) {
              struct_value.set(field, capnp::Text::Reader(kj::StringPtr(
                                          data->data(), data->size())));
            } else if (field_type.isEnum() || field_type.isUInt16()) {
              KJ_REQUIRE(data->size() == 2, data->size());
              struct_value.set(field, *(const uint16_t*)data->data());
            } else {
              KJ_FAIL_REQUIRE("Unhandled field type");
            }
          }
        }
      } break;

      default:
        KJ_FAIL_REQUIRE("Unhandled type", output.getType());
    }
  }
}

}  // namespace ev
