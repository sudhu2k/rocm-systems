# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)
include(GNUInstallDirs)

add_library(
    rocjitsu_cli_launch_preload
    STATIC
    launch_preload.cpp
    rocm_visibility.cpp
)
target_include_directories(
    rocjitsu_cli_launch_preload
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(rocjitsu_cli_launch_preload PRIVATE _GNU_SOURCE)
target_link_libraries(rocjitsu_cli_launch_preload PUBLIC ${CMAKE_DL_LIBS})
rj_configure_target(rocjitsu_cli_launch_preload TOOL)

add_library(rocjitsu_cli_json STATIC json_output.cpp)
target_include_directories(rocjitsu_cli_json PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
rj_configure_target(rocjitsu_cli_json TOOL)

add_executable(rocjitsu_bin main.cpp)

target_include_directories(
    rocjitsu_bin
    PRIVATE ${GENERATED_DIR} ${RJ_VERSION_INCLUDE_DIR}
)
target_compile_definitions(
    rocjitsu_bin
    PRIVATE
        RJ_SOURCE_CONFIG_DIR="${PROJECT_SOURCE_DIR}/configs"
        RJ_INSTALL_CONFIG_DIR="${CMAKE_INSTALL_FULL_DATADIR}/rocjitsu/configs"
        RJ_INSTALL_CONFIG_RELDIR="${CMAKE_INSTALL_DATADIR}/rocjitsu/configs"
)

add_custom_command(
    TARGET rocjitsu_bin
    POST_BUILD
    COMMAND
        ${CMAKE_COMMAND} -E copy_directory "${PROJECT_SOURCE_DIR}/configs"
        "$<TARGET_FILE_DIR:rocjitsu_bin>/../../configs"
    COMMENT "Staging rocjitsu configs for build-tree profile lookup"
)

target_link_libraries(
    rocjitsu_bin
    PRIVATE
        rocjitsu_kmd
        rocjitsu_analysis
        rocjitsu_code
        rocjitsu_code_object
        rocjitsu_executable
        rocjitsu_isa
        rocjitsu_isa_registry
        simdojo
        rocjitsu_vm
        rocjitsu_vm_amdgpu
        # Plugin loader: rj_vm_load_plugins() dlopen()s plugin modules in
        # daemon mode.
        rocjitsu_plugin_loader
        rocjitsu_vm_risc_v
        rocjitsu_config
        rocjitsu_dbt_config
        rocjitsu_ip_discovery
        util
        flatbuffers
        Threads::Threads
        rocjitsu_cli
        rocjitsu_cli_launch_preload
        rocjitsu_cli_json
        rocjitsu_version
)

# The vfio-user front end is an optional mode of this CLI, so it is the only
# consumer that pulls in the transport and its external library.
if(ROCJITSU_ENABLE_VFIO)
    target_link_libraries(
        rocjitsu_bin
        PRIVATE rocjitsu_pci rocjitsu_vmm_vfu libvfio-user::libvfio-user
    )
    target_compile_definitions(rocjitsu_bin PRIVATE RJ_ENABLE_VFIO_USER)
endif()

set_target_properties(rocjitsu_bin PROPERTIES OUTPUT_NAME rocjitsu)

rj_configure_target(rocjitsu_bin TOOL)

# Install the rocjitsu CLI so it is usable from a packaged dist as the default
# entrypoint for driving emulation.
install(TARGETS rocjitsu_bin RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(BUILD_TESTING)
    add_executable(
        rocjitsu_launch_flags_test
        ../../tests/tools/launch_flags_test.cpp
    )
    target_include_directories(
        rocjitsu_launch_flags_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(
        rocjitsu_launch_flags_test
        PRIVATE rocjitsu_cli flatbuffers GTest::gtest_main
    )
    rj_configure_target(rocjitsu_launch_flags_test TOOL)
    include(GoogleTest)
    rj_add_gtest_tests(rocjitsu_launch_flags_test TEST_PREFIX "cli.")
    if(RJ_INSTALL_TESTS)
        file(
            APPEND "${RJ_INSTALL_CTEST_FILE}"
            "include(\"rocjitsu_launch_flags_test.cmake\")\n"
        )
    endif()
endif()
