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
  set (GPGPU_CODEGEN_FILES
       CodeGen/PTXGenerator.cpp)
endif (GPU_CODEGEN)

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
    External/isl/isl_val_imath.c
    External/isl/isl_vec.c
    External/isl/isl_version.c
    External/isl/isl_vertices.c
    External/isl/print.c
    External/isl/imath/gmp_compat.c
    External/isl/imath/imath.c
    External/isl/imath/imrat.c
    External/isl/imath/iprime.c
    External/isl/imath/pi.c
    External/isl/imath/rsamath.c
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
