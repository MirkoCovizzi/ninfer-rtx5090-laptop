cmake_minimum_required(VERSION 3.28)

# The current implementation is compiled only for gfx1151 (Strix Halo RDNA 3.5).
# Set and validate the architecture before HIP compiler detection so CMake never
# injects a generic host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1151 CACHE STRING "AMD GPU architectures to build")
endif()
if(NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx1151")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_HIP_ARCHITECTURES=gfx1151; got "
    "'${CMAKE_HIP_ARCHITECTURES}'")
endif()

# CMake's HIP language wants a bare clang-based driver (it rejects the hipcc
# wrapper). Resolve the ROCm clang through hipcc's own query, then use it
# directly for C, CXX, and HIP. The unified clang toolchain is the model CMake's
# HIP support is designed around: the HIP mode flag and --offload-arch then apply
# to every source without breaking a separate host compiler.
if(NOT CMAKE_HIP_COMPILER)
  find_program(_ninfer_hipcc hipcc HINTS /opt/rocm/bin /opt/rocm/core-*/bin)
  if(NOT _ninfer_hipcc)
    message(FATAL_ERROR "hipcc not found; install ROCm and add it to PATH")
  endif()
  get_filename_component(_ninfer_hipcc_dir "${_ninfer_hipcc}" DIRECTORY)
  execute_process(COMMAND "${_ninfer_hipcc_dir}/hipconfig" -hipclangpath
                  OUTPUT_VARIABLE _ninfer_hipclang
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _ninfer_hipclang OR NOT EXISTS "${_ninfer_hipclang}")
    message(FATAL_ERROR "could not resolve the ROCm clang through hipconfig")
  endif()
  set(CMAKE_HIP_COMPILER "${_ninfer_hipclang}/clang++" CACHE FILEPATH "HIP compiler (ROCm clang++)")
  if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${_ninfer_hipclang}/clang++" CACHE FILEPATH "C++ compiler (ROCm clang++)")
  endif()
  if(NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "${_ninfer_hipclang}/clang" CACHE FILEPATH "C compiler (ROCm clang)")
  endif()
endif()

# The ROCm clang package does not ship clang-scan-deps; CMake 4.x needs it for
# the clang dependency scanner. Resolve one before compiler detection so the
# in-project search reuses it.
if(NOT CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS)
  file(GLOB _ninfer_scan_hints /usr/lib/llvm-*/bin)
  find_program(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS NAMES clang-scan-deps
    HINTS ${_ninfer_scan_hints}
    DOC "clang-scan-deps dependency scanner")
  if(NOT CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS)
    message(WARNING "clang-scan-deps not found; dependency scanning will not work")
  endif()
endif()

project(ninfer LANGUAGES C CXX HIP)

# Host (C/CXX) translation units that include project headers pull in hip_runtime.h.
# The HIP language sets the AMD platform defines for device sources; propagate the
# same contract to every language so the ROCm HIP headers are self-consistent.
add_compile_definitions(__HIP_PLATFORM_AMD__=1 __HIP_ROCclr__=1)
include_directories(BEFORE "${ROCM_PATH}/include")

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Every kernel in this engine assumes 32-lane warps (CUDA semantics). RDNA 3.5
# defaults to wave64, so force wave32 compilation globally. The gfx1151 WMMA
# instructions are only available in the wave32 form. CMake adds the HIP mode
# flag itself; only the ROCm resource path and the wave32 choice are explicit.
if(NOT DEFINED ROCM_PATH)
  get_filename_component(ROCM_PATH "${CMAKE_HIP_COMPILER}" DIRECTORY)
  get_filename_component(ROCM_PATH "${ROCM_PATH}" DIRECTORY)
  get_filename_component(ROCM_PATH "${ROCM_PATH}" DIRECTORY)
  get_filename_component(ROCM_PATH "${ROCM_PATH}" DIRECTORY)
  set(ROCM_PATH "${ROCM_PATH}" CACHE PATH "ROCm installation root")
endif()
set(CMAKE_HIP_FLAGS "--rocm-path=${ROCM_PATH} -mno-wavefrontsize64")

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS hip_link=1)
  set(CMAKE_JOB_POOL_LINK hip_link)
endif()

if(NOT CMAKE_HIP_COMPILER_VERSION VERSION_GREATER_EQUAL 6.0)
  message(FATAL_ERROR
    "NInfer requires ROCm/HIP 6.0 or newer; found HIP compiler "
    "${CMAKE_HIP_COMPILER_VERSION}")
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(HIP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
if(NINFER_BUILD_MEDIA_ACQUIRE)
  pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
endif()
if(NINFER_BUILD_SERVE)
  find_package(Threads REQUIRED)
endif()

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
