#pragma once

#include <roctracer/roctx.h>

#include <string>
#include <utility>

namespace ninfer {

class NvtxRange {
public:
    explicit NvtxRange(const char* name) { roctxRangePushA(name); }

    explicit NvtxRange(std::string name) : name_(std::move(name)) { roctxRangePushA(name_.c_str()); }

    ~NvtxRange() { roctxRangePop(); }

    NvtxRange(const NvtxRange&)            = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
    NvtxRange(NvtxRange&&)                 = delete;
    NvtxRange& operator=(NvtxRange&&)      = delete;

private:
    std::string name_;
};

} // namespace ninfer
