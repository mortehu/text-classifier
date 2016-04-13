#ifndef BASE_COLUMNFILE_CAPNP_H_
#define BASE_COLUMNFILE_CAPNP_H_ 1

#include <capnp/dynamic.h>

#include "base/columnfile.h"

namespace ev {

void WriteMessageToColumnFile(ColumnFileWriter& output, capnp::DynamicValue::Reader message);

void ReadMessageFromColumnFile(ColumnFileReader& input, capnp::DynamicValue::Builder output);

}  // namespace ev

#endif  // !BASE_COLUMNFILE_CAPNP_H_
