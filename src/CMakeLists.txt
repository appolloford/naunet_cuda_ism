set(
    OBJTARGETS
        naunet_constants
        naunet_physics
        naunet_renorm
        naunet_rates
        naunet_fex
        naunet_jac
)

# check enabled languages
get_property(
    languages
        GLOBAL
            PROPERTY ENABLED_LANGUAGES
)

# Functions from Numerical Recipes, not suppport GPU yet.
if(NOT "CUDA" IN_LIST languages)
    list(
        APPEND OBJTARGETS
            naunet_utilities
    )
endif()

foreach(name ${OBJTARGETS})
    add_library(
        ${name}
            OBJECT
            ${name}.cu
    )
    set_target_properties(
        ${name}
            PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
    )
    target_include_directories(
        ${name}
            PUBLIC
                ${CMAKE_SOURCE_DIR}/include
                ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(
        ${name}
            ${SUNDIALSLIB}
    )
endforeach()

if(MAKE_SHARED)
    add_library(
        naunet_shared
            SHARED
            naunet.cpp
    )
    set_target_properties(
        naunet_shared
            PROPERTIES
                OUTPUT_NAME naunet
    )
    set_target_properties(
        naunet_shared
            PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
    )
    target_include_directories(
        naunet_shared
            PUBLIC
                ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(
        naunet_shared
            ${OBJTARGETS}
            ${SUNDIALSLIB}
    )
    install(
        TARGETS naunet_shared
        LIBRARY
        DESTINATION lib
    )
endif()

if(MAKE_STATIC)
    add_library(
        naunet_static
            STATIC
            naunet.cpp
    )
    set_target_properties(
        naunet_static
            PROPERTIES
                OUTPUT_NAME naunet
    )
    set_target_properties(
        naunet_static
            PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
    )
    target_include_directories(
        naunet_static
            PUBLIC
                ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(
        naunet_static
            ${OBJTARGETS}
            ${SUNDIALSLIB}
    )
    install(
        TARGETS naunet_static
        LIBRARY
        DESTINATION lib
    )
endif()


if(MAKE_PYTHON)

    include(FetchContent)
    FetchContent_Declare(
        pybind11_sources
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_GetProperties(pybind11_sources)
    if(NOT pybind11_sources_POPULATED)
        FetchContent_Populate(pybind11_sources)
        add_subdirectory(
            ${pybind11_sources_SOURCE_DIR}
            ${pybind11_sources_BINARY_DIR}
        )
    endif()

    add_library(${PYMODNAME} MODULE naunet.cpp)
    target_compile_definitions(${PYMODNAME}
        PUBLIC
            -DPYMODULE
            -DPYMODNAME=${PYMODNAME}
    )

    set_target_properties(${PYMODNAME}
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            PREFIX "${PYTHON_MODULE_PREFIX}"
            SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )

    target_include_directories(
        ${PYMODNAME}
            PUBLIC
                ${CMAKE_SOURCE_DIR}/include
    )

    target_link_libraries(
        ${PYMODNAME}
            PUBLIC
                pybind11::module
            PRIVATE
                ${SUNDIALSLIB}
                ${OBJTARGETS}
    )

    install(
        TARGETS ${PYMODNAME}
        LIBRARY
        DESTINATION python
    )

endif()

install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION include
)