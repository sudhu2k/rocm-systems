# ---------------------------------------------------------------------------
# src/tuning/: one binary per test file, each linking only the fakes its unit needs.
# ---------------------------------------------------------------------------
set(_MICRO_TUNING_DEFS
  # graph/tuning.cc keeps the basename, so add_file_unique() renames this one.
  TUNING_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/tuning/tuning_tmp.cc"
  TUNING_GENERAL_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/tuning/tuning_general.cc"
  SYM_MODEL_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/tuning/sym_model.cc"
  # gin_tmp.cc, not gin.cc: src/CMakeLists.txt lists plugin/gin.cc first, so hipify renames this one.
  SYM_MODEL_GIN_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/tuning/sym_model/gin_tmp.cc"
  SYM_MODEL_LSA_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/tuning/sym_model/lsa.cc"
  SYM_MODEL_LSA_BASE_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/tuning/sym_model/lsa_base.cc"
  SYM_MODEL_LSA_A2A_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/tuning/sym_model/lsa_a2a.cc")

# One src/tuning/ test file per binary; see rccl_add_unit_micro_binary().
macro(rccl_add_tuning_micro_binary _name _test)
  rccl_add_unit_micro_binary(${_name} tuning/${_test} ${ARGN}
    OBJECT_PREFIX rccl-tuning-obj
    COMMON_INCLUDES
      ${RCCL_HOST_HIPIFY_DIR}/src/device  # sym_kernels.cc includes "tdm/tdmCopy.h" (SymModel)
    COMMON_DEFS ${_MICRO_TUNING_DEFS}
    CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/../test_categories_micro_tuning.yaml")
endmacro()

rccl_add_tuning_micro_binary(rccl-UnitTestsMicroTuning tuning-test.cc
  SOURCES
    fakes/collectives_fakes.cc
    fakes/env_fakes.cc
    fakes/group_fakes.cc
    fakes/nccl_fakes.cc)

rccl_add_tuning_micro_binary(rccl-UnitTestsMicroTuningGeneral tuning-general-test.cc
  SOURCES
    fakes/env_fakes.cc
    fakes/nccl_fakes.cc)

# sym_kernels_host.cc only exists when sym kernels are generated.
if(_MICRO_BUILD_SYM_KERNELS)
  rccl_add_tuning_micro_binary(rccl-UnitTestsMicroTuningSymModel sym-model-test.cc
    SOURCES
      tuning/sym_kernels_wrapper.cc
      fakes/nccl_fakes.cc
      fakes/env_fakes.cc
      fakes/devcomm_fakes.cc
      fakes/nccl_device_core_fakes.cc      # ncclTeamRail, read by the GIN cost model
      ${RCCL_HOST_HIPIFY_DIR}/src/misc/archinfo.cc
    EXTRA_DEFS
      ${_MICRO_AMDSMI_DEF}
      GENERATE_SYM_KERNELS
      SYM_KERNELS_HOST_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/gensrc/symmetric/sym_kernels_host.cc"
      SYM_KERNELS_CC_PATH="${RCCL_HOST_HIPIFY_DIR}/src/sym_kernels.cc")
endif()
