# To use CUDA by enabling language
cmake_minimum_required(VERSION 3.18)
project(NAUNET VERSION 0.4.2 LANGUAGES CUDA CXX)

include(CMakePrintHelpers)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DNAUNET_DEBUG)
endif()

message(STATUS "CMAKE_SYSTEM_NAME          = ${CMAKE_SYSTEM_NAME}"         )
message(STATUS "CMAKE_CXX_COMPILER_ID      = ${CMAKE_CXX_COMPILER_ID}"     )
message(STATUS "CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_COMPILER_IS_GNUCXX   = ${CMAKE_COMPILER_IS_GNUCXX}"  )
message(STATUS "CMAKE_BUILD_TYPE           = ${CMAKE_BUILD_TYPE}"          )

set(SUNDIALS_MIN_VERSION "6.0.0")
set(SUNDIALS_DIR "" CACHE PATH "Path to SUNDIALSConfig.cmake")
find_package(SUNDIALS ${SUNDIALS_MIN_VERSION} REQUIRED CONFIG)

message(STATUS "SUNDIALS_FOUND             = ${SUNDIALS_FOUND}")


list(APPEND SUNDIALSLIB
    SUNDIALS::cvode
    SUNDIALS::nveccuda
    SUNDIALS::sunmatrixcusparse
    SUNDIALS::sunlinsolcusolversp
    )

# find_package(CUDA REQUIRED)
# find_package(CUDAToolkit REQUIRED)

# if(CMAKE_BUILD_TYPE STREQUAL Debug)
#     cmake_print_properties(TARGETS SUNDIALS::cvode_static PROPERTIES
#                            LOCATION INTERFACE_INCLUDE_DIRECTORIES)
#     cmake_print_properties(TARGETS SUNDIALS::nvecserial_static PROPERTIES
#                            LOCATION INTERFACE_INCLUDE_DIRECTORIES)
# endif()

set(NVIDIA_GPU_COUNT 0)

# check the available GPUs if using GPU solver
find_program(_nvidia_smi NAMES nvidia-smi)
if (_nvidia_smi)
    set(DETECTED_GPU_COUNT_NVIDIA_SMI 0)

    # execute nvidia-smi -L to get a short list of GPUs available
    exec_program(${_nvidia_smi} ARGS -L
        OUTPUT_VARIABLE _nvidia_smi_out
        RETURN_VALUE    _nvidia_smi_ret)

    # process the stdout of nvidia-smi
    if (_nvidia_smi_ret EQUAL 0)
        # convert string with newlines to list of strings
        string(REGEX REPLACE "\n" ";" _nvidia_smi_out "${_nvidia_smi_out}")
        foreach(_line ${_nvidia_smi_out})
            if (_line MATCHES "^GPU [0-9]+:")
                math(EXPR DETECTED_GPU_COUNT_NVIDIA_SMI "${DETECTED_GPU_COUNT_NVIDIA_SMI}+1")
                # the UUID is not very useful for the user, remove it
                string(REGEX REPLACE " \\(UUID:.*\\)" "" _gpu_info "${_line}")
                if (NOT _gpu_info STREQUAL "")
                    list(APPEND DETECT_GPU_INFO "${_gpu_info}")
                endif()
            endif()
        endforeach()
    endif()

    set(NVIDIA_GPU_COUNT ${DETECTED_GPU_COUNT_NVIDIA_SMI})
endif()

message(STATUS "NVIDIA_GPU_COUNT           = " ${NVIDIA_GPU_COUNT})

set(MAKE_STATIC ON CACHE BOOL "")
set(MAKE_SHARED ON CACHE BOOL "")
set(MAKE_PYTHON OFF CACHE BOOL "")
set(MAKE_TEST ON CACHE BOOL "")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MAKE_PYTHON)
    set(PYMODNAME "pynaunet" CACHE STRING "")
    find_package(Python REQUIRED COMPONENTS Interpreter Development)
endif()

add_subdirectory(src)

if(MAKE_TEST)
    enable_testing()
    add_subdirectory(tests)
endif()
