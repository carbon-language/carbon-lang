# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A Starlark cc_toolchain configuration rule"""

load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "tool",
    "tool_path",
    "variable_with_value",
    "with_feature_set",
)
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load(
    ":clang_detected_variables.bzl",
    "clang_bindir",
    "clang_version",
    "clang_version_for_cache",
    "clang_include_dirs_list",
    "clang_resource_dir",
    "llvm_bindir",
    "sysroot_dir",
)

all_c_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.assemble,
    ACTION_NAMES.preprocess_assemble,
]

all_cpp_compile_actions = [
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
    ACTION_NAMES.cpp_module_codegen,
]

all_compile_actions = all_c_compile_actions + all_cpp_compile_actions

preprocessor_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_header_parsing,
    ACTION_NAMES.cpp_module_compile,
]

codegen_compile_actions = [
    ACTION_NAMES.c_compile,
    ACTION_NAMES.cpp_compile,
    ACTION_NAMES.linkstamp_compile,
    ACTION_NAMES.assemble,
    ACTION_NAMES.preprocess_assemble,
    ACTION_NAMES.cpp_module_codegen,
]

all_link_actions = [
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

def _impl(ctx):
    tool_paths = [
        tool_path(name = "ar", path = llvm_bindir + "/llvm-ar"),
        tool_path(name = "ld", path = clang_bindir + "/ld.lld"),
        tool_path(name = "cpp", path = clang_bindir + "/clang-cpp"),
        tool_path(name = "gcc", path = clang_bindir + "/clang++"),
        tool_path(name = "dwp", path = llvm_bindir + "/llvm-dwp"),
        tool_path(name = "gcov", path = llvm_bindir + "/llvm-cov"),
        tool_path(name = "nm", path = llvm_bindir + "/llvm-nm"),
        tool_path(name = "objcopy", path = llvm_bindir + "/llvm-objcopy"),
        tool_path(name = "objdump", path = llvm_bindir + "/llvm-objdump"),
        tool_path(name = "strip", path = llvm_bindir + "/llvm-strip"),
    ]

    action_configs = [
        action_config(action_name = name, enabled = True, tools = [tool(path = clang_bindir + "/clang")])
        for name in all_c_compile_actions
    ] + [
        action_config(action_name = name, enabled = True, tools = [tool(path = clang_bindir + "/clang++")])
        for name in all_cpp_compile_actions
    ] + [
        action_config(action_name = name, enabled = True, tools = [tool(path = clang_bindir + "/clang++")])
        for name in all_link_actions
    ] + [
        action_config(action_name = name, enabled = True, tools = [tool(path = llvm_bindir + "/llvm-ar")])
        for name in [ACTION_NAMES.cpp_link_static_library]
    ] + [
        action_config(action_name = name, enabled = True, tools = [tool(path = llvm_bindir + "/llvm-strip")])
        for name in [ACTION_NAMES.strip]
    ]

    std_compile_flags = ["-std=c++17"]
    # libc++ is only used on non-Windows platforms.
    if ctx.attr.target_cpu != "x64_windows":
        std_compile_flags.append("-stdlib=libc++")

    default_flags_feature = feature(
        name = "default_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_compile_actions + all_link_actions,
                flag_groups = ([
                    flag_group(
                        flags = [
                            "-no-canonical-prefixes",
                            "-fcolor-diagnostics",
                        ],
                    ),
                ]),
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = ([
                    flag_group(
                        flags = [
                            "-Werror",
                            "-Wall",
                            "-Wextra",
                            "-Wthread-safety",
                            "-Wself-assign",
                            "-Wimplicit-fallthrough",
                            "-Wctad-maybe-unsupported",
                            "-Wnon-virtual-dtor",
                            # Unfortunately, LLVM isn't clean for this warning.
                            "-Wno-unused-parameter",
                            # Compile actions shouldn't link anything.
                            "-c",
                        ],
                    ),
                    flag_group(
                        expand_if_available = "output_assembly_file",
                        flags = ["-S"],
                    ),
                    flag_group(
                        expand_if_available = "output_preprocess_file",
                        flags = ["-E"],
                    ),
                    flag_group(
                        flags = ["-MD", "-MF", "%{dependency_file}"],
                        expand_if_available = "dependency_file",
                    ),
                    flag_group(
                        flags = ["-frandom-seed=%{output_file}"],
                        expand_if_available = "output_file",
                    ),
                ]),
            ),
            flag_set(
                actions = all_cpp_compile_actions + all_link_actions,
                flag_groups = ([
                    flag_group(
                        flags = std_compile_flags,
                    ),
                ]),
            ),
            flag_set(
                actions = codegen_compile_actions,
                flag_groups = ([
                    flag_group(
                        flags = [
                            "-ffunction-sections",
                            "-fdata-sections",
                        ],
                    ),
                ]),
            ),
            flag_set(
                actions = codegen_compile_actions,
                flag_groups = [
                    flag_group(flags = ["-fPIC"], expand_if_available = "pic"),
                ],
            ),
            flag_set(
                actions = preprocessor_compile_actions,
                flag_groups = [
                    flag_group(
                        flags = [
                            # Disable a warning and override builtin macros to
                            # ensure a hermetic build.
                            "-Wno-builtin-macro-redefined",
                            "-D__DATE__=\"redacted\"",
                            "-D__TIMESTAMP__=\"redacted\"",
                            "-D__TIME__=\"redacted\"",
                            # Pass the clang version as a define so that bazel
                            # caching is more likely to notice version changes.
                            "-DCLANG_VERSION_FOR_CACHE=\"%s\"" % clang_version_for_cache,
                        ],
                    ),
                    flag_group(
                        flags = ["-D%{preprocessor_defines}"],
                        iterate_over = "preprocessor_defines",
                    ),
                    flag_group(
                        flags = ["-include", "%{includes}"],
                        iterate_over = "includes",
                        expand_if_available = "includes",
                    ),
                    flag_group(
                        flags = ["-iquote", "%{quote_include_paths}"],
                        iterate_over = "quote_include_paths",
                    ),
                    flag_group(
                        flags = ["-I%{include_paths}"],
                        iterate_over = "include_paths",
                    ),
                    flag_group(
                        flags = ["-isystem", "%{system_include_paths}"],
                        iterate_over = "system_include_paths",
                    ),
                ],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(flags = ["-shared"])],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["-Wl,--gdb-index"],
                        expand_if_available = "is_using_fission",
                    ),
                    flag_group(
                        flags = ["-Wl,-S"],
                        expand_if_available = "strip_debug_symbols",
                    ),
                    flag_group(
                        flags = ["-L%{library_search_directories}"],
                        iterate_over = "library_search_directories",
                        expand_if_available = "library_search_directories",
                    ),
                    flag_group(
                        iterate_over = "runtime_library_search_directories",
                        flags = [
                            "-Wl,-rpath,$ORIGIN/%{runtime_library_search_directories}",
                        ],
                        expand_if_available =
                            "runtime_library_search_directories",
                    ),
                ],
            ),
        ],
    )

    # Handle different levels of optimization with individual features so that
    # they can be ordered and the defaults can override the minimal settings if
    # both are enabled.
    minimal_optimization_flags = feature(
        name = "minimal_optimization_flags",
        flag_sets = [
            flag_set(
                actions = codegen_compile_actions,
                flag_groups = [flag_group(flags = [
                    "-O1",
                ])],
            ),
            # Use a conditional flag set for enabling the fast instruction
            # selector to work around an LLVM bug:
            # https://github.com/llvm/llvm-project/issues/56133
            flag_set(
                actions = codegen_compile_actions,
                flag_groups = [flag_group(flags = [
                    "-mllvm",
                    "-fast-isel",
                ])],
                with_features = [
                    with_feature_set(not_features = ["fuzzer"]),
                ],
            ),
        ],
    )
    default_optimization_flags = feature(
        name = "default_optimization_flags",
        enabled = True,
        requires = [feature_set(["opt"])],
        flag_sets = [
            flag_set(
                actions = all_compile_actions,
                flag_groups = [flag_group(flags = [
                    "-DNDEBUG",
                ])],
            ),
            flag_set(
                actions = codegen_compile_actions,
                flag_groups = [flag_group(flags = [
                    "-O3",
                ])],
            ),
        ],
    )

    # Handle different levels and forms of debug info emission with individual
    # features so that they can be ordered and the defaults can override the
    # minimal settings if both are enabled.
    minimal_debug_info_flags = feature(
        name = "minimal_debug_info_flags",
        flag_sets = [
            flag_set(
                actions = codegen_compile_actions,
                flag_groups = [
                    flag_group(
                        flags = ["-gmlt"],
                    ),
                ],
            ),
        ],
    )
    default_debug_info_flags = feature(
        name = "default_debug_info_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = codegen_compile_actions,
                flag_groups = ([
                    flag_group(
                        flags = ["-g"],
                    ),
                ]),
                with_features = [with_feature_set(features = ["dbg"])],
            ),
            flag_set(
                actions = codegen_compile_actions,
                flag_groups = [
                    flag_group(
                        flags = ["-gsplit-dwarf", "-g"],
                        expand_if_available = "per_object_debug_info_file",
                    ),
                ],
            ),
        ],
    )

    # This feature can be enabled in conjunction with any optimizations to
    # ensure accurate call stacks and backtraces for profilers or errors.
    preserve_call_stacks = feature(
        name = "preserve_call_stacks",
        flag_sets = [flag_set(
            actions = codegen_compile_actions,
            flag_groups = [flag_group(flags = [
                # Ensure good backtraces by preserving frame pointers and
                # disabling tail call elimination.
                "-fno-omit-frame-pointer",
                "-mno-omit-leaf-frame-pointer",
                "-fno-optimize-sibling-calls",
            ])],
        )],
    )

    sysroot_feature = feature(
        name = "sysroot",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_compile_actions + all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["--sysroot=%{sysroot}"],
                        expand_if_available = "sysroot",
                    ),
                ],
            ),
        ],
    )

    use_module_maps = feature(
        name = "use_module_maps",
        requires = [feature_set(features = ["module_maps"])],
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [
                    # These flag groups are separate so they do not expand to
                    # the cross product of the variables.
                    flag_group(flags = ["-fmodule-name=%{module_name}"]),
                    flag_group(
                        flags = ["-fmodule-map-file=%{module_map_file}"],
                    ),
                ],
            ),
        ],
    )

    # Tell bazel we support module maps in general, so they will be generated
    # for all c/c++ rules.
    # Note: not all C++ rules support module maps; thus, do not imply this
    # feature from other features - instead, require it.
    module_maps = feature(
        name = "module_maps",
        enabled = True,
        implies = [
            # "module_map_home_cwd",
            # "module_map_without_extern_module",
            # "generate_submodules",
        ],
    )

    layering_check = feature(
        name = "layering_check",
        implies = ["use_module_maps"],
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [
                    flag_group(flags = [
                        "-fmodules-strict-decluse",
                        "-Wprivate-header",
                    ]),
                    flag_group(
                        iterate_over = "dependent_module_map_files",
                        flags = [
                            "-fmodule-map-file=%{dependent_module_map_files}",
                        ],
                    ),
                ],
            ),
        ],
    )

    sanitizer_common_flags = feature(
        name = "sanitizer_common_flags",
        requires = [feature_set(["nonhost"])],
        implies = ["minimal_optimization_flags", "minimal_debug_info_flags", "preserve_call_stacks"],
        flag_sets = [flag_set(
            actions = all_link_actions,
            flag_groups = [flag_group(flags = [
                "-static-libsan",
            ])],
        )],
    )

    asan = feature(
        name = "asan",
        requires = [feature_set(["nonhost"])],
        implies = ["sanitizer_common_flags"],
        flag_sets = [flag_set(
            actions = all_compile_actions + all_link_actions,
            flag_groups = [flag_group(flags = [
                "-fsanitize=address,undefined,nullability",
                "-fsanitize-address-use-after-scope",
                # We don't need the recovery behavior of UBSan as we expect
                # builds to be clean. Not recovering is a bit cheaper.
                "-fno-sanitize-recover=undefined",
                # Don't embed the full path name for files. This limits the size
                # and combined with line numbers is unlikely to result in many
                # ambiguities.
                "-fsanitize-undefined-strip-path-components=-1",
                # Needed due to clang AST issues, such as in
                # clang/AST/Redeclarable.h line 199.
                "-fno-sanitize=vptr",
            ])],
        )],
    )

    enable_asan_in_fastbuild = feature(
        name = "enable_asan_in_fastbuild",
        enabled = True,
        requires = [feature_set(["nonhost", "fastbuild"])],
        implies = ["asan"],
    )

    fuzzer = feature(
        name = "fuzzer",
        requires = [feature_set(["nonhost"])],
        implies = ["asan"],
        flag_sets = [flag_set(
            actions = all_compile_actions + all_link_actions,
            flag_groups = [flag_group(flags = [
                "-fsanitize=fuzzer-no-link",
            ])],
        )],
    )

    proto_fuzzer = feature(
        name = "proto-fuzzer",
        enabled = False,
        requires = [feature_set(["nonhost"])],
        implies = ["fuzzer"],
    )

    # With clang 14 and lower, we expect it to be built with libc++ debug
    # support. In later LLVM versions, we expect the assertions define to work.
    if clang_version and clang_version <= 14:
        libcpp_debug_flags = ["-D_LIBCPP_DEBUG=1"]
    else:
        libcpp_debug_flags = ["-D_LIBCPP_ENABLE_ASSERTIONS=1"]

    linux_flags_feature = feature(
        name = "linux_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = ([
                    flag_group(
                        flags = [
                            "-fuse-ld=lld",
                            "-stdlib=libc++",
                            "-unwindlib=libunwind",
                            # Force the C++ standard library and runtime
                            # libraries to be statically linked. This works even
                            # with libc++ and libunwind despite the names,
                            # provided libc++ is built with two CMake options:
                            # - `-DCMAKE_POSITION_INDEPENDENT_CODE=ON`
                            # - `-DLIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY`
                            # These are both required because of PR43604
                            # (impacting at least Debian packages of libc++) and
                            # PR46321 (impacting most other packages).
                            # We recommend using Homebrew's LLVM install on
                            # Linux.
                            "-static-libstdc++",
                            "-static-libgcc",
                            # Link with Clang's runtime library. This is always
                            # linked statically.
                            "-rtlib=compiler-rt",
                            # Explicitly add LLVM libs to the search path to
                            # preempt the detected GCC installation's library
                            # paths. Those might have a system installed libc++
                            # and we want to find the one next to our Clang.
                            "-L" + llvm_bindir + "/../lib",
                            # Link with pthread.
                            "-lpthread",
                        ],
                    ),
                ]),
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = [flag_group(flags = libcpp_debug_flags)],
                with_features = [
                    with_feature_set(not_features = ["opt"]),
                ],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-pie"],
                        expand_if_available = "force_pic",
                    ),
                ],
            ),
        ],
    )

    macos_flags_feature = feature(
        name = "macos_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-fpie"],
                        expand_if_available = "force_pic",
                    ),
                ],
            ),
        ],
    )

    default_link_libraries_feature = feature(
        name = "default_link_libraries",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["%{linkstamp_paths}"],
                        iterate_over = "linkstamp_paths",
                        expand_if_available = "linkstamp_paths",
                    ),
                    flag_group(
                        iterate_over = "libraries_to_link",
                        flag_groups = [
                            flag_group(
                                flags = ["-Wl,--start-lib"],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file_group",
                                ),
                            ),
                            flag_group(
                                flags = ["-Wl,-whole-archive"],
                                expand_if_true =
                                    "libraries_to_link.is_whole_archive",
                            ),
                            flag_group(
                                flags = ["%{libraries_to_link.object_files}"],
                                iterate_over = "libraries_to_link.object_files",
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file_group",
                                ),
                            ),
                            flag_group(
                                flags = ["%{libraries_to_link.name}"],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file",
                                ),
                            ),
                            flag_group(
                                flags = ["%{libraries_to_link.name}"],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "interface_library",
                                ),
                            ),
                            flag_group(
                                flags = ["%{libraries_to_link.name}"],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "static_library",
                                ),
                            ),
                            flag_group(
                                flags = ["-l%{libraries_to_link.name}"],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "dynamic_library",
                                ),
                            ),
                            flag_group(
                                flags = ["-l:%{libraries_to_link.name}"],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "versioned_dynamic_library",
                                ),
                            ),
                            flag_group(
                                flags = ["-Wl,-no-whole-archive"],
                                expand_if_true = "libraries_to_link.is_whole_archive",
                            ),
                            flag_group(
                                flags = ["-Wl,--end-lib"],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file_group",
                                ),
                            ),
                        ],
                        expand_if_available = "libraries_to_link",
                    ),
                    # Note that the params file comes at the end, after the
                    # libraries to link above.
                    flag_group(
                        expand_if_available = "linker_param_file",
                        flags = ["@%{linker_param_file}"],
                    ),
                ],
            ),
        ],
    )

    # Place user provided compile flags after all the features so that these
    # flags can override or customize behavior. The only thing user flags
    # cannot override is the output file as Bazel depends on that.
    #
    # Finally, place the source file (if present) and output file last to make
    # reading the compile command lines easier for humans.
    final_flags_feature = feature(
        name = "final_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_compile_actions,
                flag_groups = [
                    flag_group(
                        flags = ["%{user_compile_flags}"],
                        iterate_over = "user_compile_flags",
                        expand_if_available = "user_compile_flags",
                    ),
                    flag_group(
                        flags = ["%{source_file}"],
                        expand_if_available = "source_file",
                    ),
                    flag_group(
                        expand_if_available = "output_file",
                        flags = ["-o", "%{output_file}"],
                    ),
                ],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["%{user_link_flags}"],
                        iterate_over = "user_link_flags",
                        expand_if_available = "user_link_flags",
                    ),
                    flag_group(
                        flags = ["-o", "%{output_execpath}"],
                        expand_if_available = "output_execpath",
                    ),
                ],
            ),
        ],
    )

    # Archive actions have an entirely independent set of flags and don't
    # interact with either compiler or link actions.
    default_archiver_flags_feature = feature(
        name = "default_archiver_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(flags = ["rcsD"]),
                    flag_group(
                        flags = ["%{output_execpath}"],
                        expand_if_available = "output_execpath",
                    ),
                    flag_group(
                        iterate_over = "libraries_to_link",
                        flag_groups = [
                            flag_group(
                                flags = ["%{libraries_to_link.name}"],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file",
                                ),
                            ),
                            flag_group(
                                flags = ["%{libraries_to_link.object_files}"],
                                iterate_over = "libraries_to_link.object_files",
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file_group",
                                ),
                            ),
                        ],
                        expand_if_available = "libraries_to_link",
                    ),
                    flag_group(
                        expand_if_available = "linker_param_file",
                        flags = ["@%{linker_param_file}"],
                    ),
                ],
            ),
        ],
    )

    # Now that we have built up the constituent feature definitions, compose
    # them, including configuration based on the target platform. Currently,
    # the target platform is configured with the "cpu" attribute for legacy
    # reasons. Further, for legacy reasons the default is a Linux OS target and
    # the x88-64 CPU name is "k8".

    # First, define features that are simply used to configure others.
    features = [
        feature(name = "dbg"),
        feature(name = "fastbuild"),
        feature(name = "host"),
        feature(name = "no_legacy_features"),
        feature(name = "nonhost"),
        feature(name = "opt"),
        feature(name = "supports_dynamic_linker", enabled = ctx.attr.target_cpu == "k8"),
        feature(name = "supports_pic", enabled = True),
        feature(name = "supports_start_end_lib", enabled = ctx.attr.target_cpu == "k8"),
    ]

    # The order of the features determines the relative order of flags used.
    # Start off adding the baseline features.
    features += [
        default_flags_feature,
        minimal_optimization_flags,
        default_optimization_flags,
        minimal_debug_info_flags,
        default_debug_info_flags,
        preserve_call_stacks,
        sysroot_feature,
        sanitizer_common_flags,
        asan,
        enable_asan_in_fastbuild,
        fuzzer,
        proto_fuzzer,
        layering_check,
        module_maps,
        use_module_maps,
        default_archiver_flags_feature,
    ]

    # Next, add the features based on the target platform. Here too the
    # features are order sensitive. We also setup the sysroot here.
    if ctx.attr.target_cpu == "k8":
        features += [linux_flags_feature]
        sysroot = None
    elif ctx.attr.target_cpu == "x64_windows":
        # TODO: Need to figure out if we need to add windows specific features
        # I think the .pdb debug files will need to be handled differently,
        # so that might be an example where a feature must be added.
        sysroot = None
    elif ctx.attr.target_cpu in ["darwin", "darwin_arm64"]:
        features += [macos_flags_feature]
        sysroot = sysroot_dir
    else:
        fail("Unsupported target platform!")

    # Finally append the libraries to link and any final flags.
    features += [
        default_link_libraries_feature,
        final_flags_feature,
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        action_configs = action_configs,
        cxx_builtin_include_directories = clang_include_dirs_list + [
            # Add Clang's resource directory to the end of the builtin include
            # directories to cover the use of sanitizer resource files by the driver.
            clang_resource_dir + "/share",
        ],
        builtin_sysroot = sysroot,

        # This configuration only supports local non-cross builds so derive
        # everything from the target CPU selected.
        toolchain_identifier = "local-" + ctx.attr.target_cpu,
        host_system_name = "local-" + ctx.attr.target_cpu,
        target_system_name = "local-" + ctx.attr.target_cpu,
        target_cpu = ctx.attr.target_cpu,

        # These attributes aren't meaningful at all so just use placeholder
        # values.
        target_libc = "local",
        compiler = "local",
        abi_version = "local",
        abi_libc_version = "local",

        # We do have to pass in our tool paths.
        tool_paths = tool_paths,
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
        "target_cpu": attr.string(mandatory = True),
    },
    provides = [CcToolchainConfigInfo],
)
