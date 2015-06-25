set(LLVM_NO_RTTI 1)

set(POLLY_JSON_FILES
    JSON/json_reader.cpp
    JSON/json_value.cpp
    JSON/json_writer.cpp
)

set(ISL_CODEGEN_FILES
    CodeGen/IslAst.cpp
    CodeGen/IslExprBuilder.cpp
    CodeGen/IslNodeBuilder.cpp
    CodeGen/CodeGeneration.cpp)

if (GPU_CODEGEN)
  set (GPGPU_CODEGEN_FILES)
endif (GPU_CODEGEN)


# External: Integer Set Library
set(ISL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/External/isl")
set(ISL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/External/isl")

# Determine version of isl
if (EXISTS "${ISL_SOURCE_DIR}/GIT_HEAD_ID")
  # The source comes from a 'make dist' archive
  file(READ "${ISL_SOURCE_DIR}/GIT_HEAD_ID" GIT_HEAD_ID)
  string(STRIP "${GIT_HEAD_ID}" GIT_HEAD_ID)
elseif (EXISTS "${ISL_SOURCE_DIR}/gitversion.h")
  # The source directory is preconfigured
  file(READ "${ISL_SOURCE_DIR}/gitversion.h" GITVERSION_H)
  string(REGEX REPLACE ".*\\\"([^\\\"]*)\\\".*" "\\1" GIT_HEAD_ID "${GITVERSION_H}")
elseif ()
  # Unknown revision
  set(GIT_HEAD_ID "UNKNOWN")
endif ()

message(STATUS "ISL version: ${GIT_HEAD_ID}")

# Enable small integer optimization and imath
set(USE_GMP_FOR_MP OFF)
set(USE_IMATH_FOR_MP ON)
set(USE_SMALL_INT_OPT ON)

# Determine compiler characteristics
include(CheckCSourceCompiles)

check_c_source_compiles("
  int func(void) __attribute__((__warn_unused_result__));
  int main() { return 0; }
  " HAS_ATTRIBUTE_WARN_UNUSED_RESULT)
set(GCC_WARN_UNUSED_RESULT)
if (HAS_ATTRIBUTE_WARN_UNUSED_RESULT)
  set(GCC_WARN_UNUSED_RESULT "__attribute__((__warn_unused_result__))")
endif ()

check_c_source_compiles("
  #include <strings.h>
  int main() { ffs(0); return 0; }
  " HAVE_DECL_FFS)
if (NOT HAVE_DECL_FFS)
  set(HAVE_DECL_FFS 0)
endif ()

check_c_source_compiles("
  int main() { __builtin_ffs(0); return 0; }
  " HAVE_DECL___BUILTIN_FFS)
if (NOT HAVE_DECL___BUILTIN_FFS)
  set(HAVE_DECL___BUILTIN_FFS 0)
endif ()

check_c_source_compiles("
  static void foo(void) __attribute__ ((unused));
  int main() { return 0; }
  " HAVE___ATTRIBUTE__)

# Write configure result
configure_file("External/gitversion.h.cmake" "${ISL_BINARY_DIR}/gitversion.h")
configure_file("External/isl_config.h.cmake" "${ISL_BINARY_DIR}/isl_config.h")
file(WRITE "${ISL_BINARY_DIR}/include/isl/stdint.h" "#include <stdint.h>")

# ISL files to compile
set (ISL_FILES
    External/isl/basis_reduction_tab.c
    External/isl/isl_aff.c
    External/isl/isl_affine_hull.c
    External/isl/isl_arg.c
    External/isl/isl_ast_build.c
    External/isl/isl_ast_build_expr.c
    External/isl/isl_ast.c
    External/isl/isl_ast_codegen.c
    External/isl/isl_ast_graft.c
    External/isl/isl_band.c
    External/isl/isl_bernstein.c
    External/isl/isl_blk.c
    External/isl/isl_bound.c
    External/isl/isl_coalesce.c
    External/isl/isl_constraint.c
    External/isl/isl_convex_hull.c
    External/isl/isl_ctx.c
    External/isl/isl_deprecated.c
    External/isl/isl_dim_map.c
    External/isl/isl_equalities.c
    External/isl/isl_factorization.c
    External/isl/isl_farkas.c
    External/isl/isl_flow.c
    External/isl/isl_fold.c
    External/isl/isl_hash.c
    External/isl/isl_id.c
    External/isl/isl_id_to_ast_expr.c
    External/isl/isl_id_to_pw_aff.c
    External/isl/isl_ilp.c
    External/isl/isl_imath.c
    External/isl/isl_input.c
    External/isl/isl_int_sioimath.c
    External/isl/isl_local_space.c
    External/isl/isl_lp.c
    External/isl/isl_map.c
    External/isl/isl_map_list.c
    External/isl/isl_map_simplify.c
    External/isl/isl_map_subtract.c
    External/isl/isl_map_to_basic_set.c
    External/isl/isl_mat.c
    External/isl/isl_morph.c
    External/isl/isl_obj.c
    External/isl/isl_options.c
    External/isl/isl_output.c
    External/isl/isl_point.c
    External/isl/isl_polynomial.c
    External/isl/isl_printer.c
    External/isl/isl_range.c
    External/isl/isl_reordering.c
    External/isl/isl_sample.c
    External/isl/isl_scan.c
    External/isl/isl_schedule.c
    External/isl/isl_schedule_band.c
    External/isl/isl_schedule_node.c
    External/isl/isl_schedule_read.c
    External/isl/isl_schedule_tree.c
    External/isl/isl_scheduler.c
    External/isl/isl_seq.c
    External/isl/isl_set_list.c
    External/isl/isl_sort.c
    External/isl/isl_space.c
    External/isl/isl_stream.c
    External/isl/isl_tab.c
    External/isl/isl_tab_pip.c
    External/isl/isl_tarjan.c
    External/isl/isl_transitive_closure.c
    External/isl/isl_union_map.c
    External/isl/isl_val.c
    External/isl/isl_val_sioimath.c
    External/isl/isl_vec.c
    External/isl/isl_version.c
    External/isl/isl_vertices.c
    External/isl/print.c
    External/isl/imath/gmp_compat.c
    External/isl/imath/imath.c
    External/isl/imath/imrat.c
    )


add_polly_library(Polly
  Analysis/DependenceInfo.cpp
  Analysis/ScopDetection.cpp
  Analysis/ScopDetectionDiagnostic.cpp
  Analysis/ScopInfo.cpp
  Analysis/ScopGraphPrinter.cpp
  Analysis/ScopPass.cpp
  Analysis/TempScopInfo.cpp
  CodeGen/BlockGenerators.cpp
  ${ISL_CODEGEN_FILES}
  CodeGen/LoopGenerators.cpp
  CodeGen/IRBuilder.cpp
  CodeGen/Utils.cpp
  CodeGen/RuntimeDebugBuilder.cpp
  ${GPGPU_CODEGEN_FILES}
  Exchange/JSONExporter.cpp
  Support/GICHelper.cpp
  Support/SCEVValidator.cpp
  Support/RegisterPasses.cpp
  Support/ScopHelper.cpp
  Support/ScopLocation.cpp
  ${POLLY_JSON_FILES}
  Transform/Canonicalization.cpp
  Transform/CodePreparation.cpp
  Transform/DeadCodeElimination.cpp
  Transform/IndependentBlocks.cpp
  Transform/ScheduleOptimizer.cpp
  ${ISL_FILES}
  )

# ISL requires at least C99 to compile. gcc < 5.0 use -std=gnu89 as default.
target_enable_c99(Polly)

if (BUILD_SHARED_LIBS)
  target_link_libraries(Polly
    LLVMSupport
    LLVMCore
    LLVMScalarOpts
    LLVMInstCombine
    LLVMTransformUtils
    LLVMAnalysis
    LLVMipo
    LLVMMC
  )
  link_directories(
    ${LLVM_LIBRARY_DIR}
  )
endif()

# Build a monolithic Polly.a and a thin module LLVMPolly.moduleext that links to
# that static library.
add_polly_loadable_module(LLVMPolly
  Polly.cpp
)

if (TARGET intrinsics_gen)
  # Check if we are building as part of an LLVM build
  add_dependencies(Polly intrinsics_gen)
endif()

target_link_libraries(LLVMPolly Polly)

set_target_properties(LLVMPolly
  PROPERTIES
  LINKER_LANGUAGE CXX
  PREFIX "")
