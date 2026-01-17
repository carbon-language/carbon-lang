# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Definitions of general C++ `cc_toolchain_config` features."""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load(
    "@rules_cc//cc:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
    "with_feature_set",
)
load(
    ":cc_toolchain_actions.bzl",
    "all_compile_actions",
    "all_cpp_compile_actions",
    "all_link_actions",
    "codegen_compile_actions",
    "preprocessor_compile_actions",
)

clang_feature = feature(
    name = "clang",
    enabled = True,
    flag_sets = [
        flag_set(
            actions = all_compile_actions + all_link_actions,
            flag_groups = [
                flag_group(flags = [
                    "-no-canonical-prefixes",
                    "-fcolor-diagnostics",
                ]),
                flag_group(
                    expand_if_available = "sysroot",
                    flags = ["--sysroot=%{sysroot}"],
                ),
            ],
        ),
        flag_set(
            actions = all_compile_actions,
            flag_groups = [
                flag_group(flags = [
                    # Compile actions shouldn't link anything.
                    "-c",
                ]),

                # Flags controlling the production of specific outputs from
                # compile actions.
                flag_group(
                    expand_if_available = "output_assembly_file",
                    flags = ["-S"],
                ),
                flag_group(
                    expand_if_available = "output_preprocess_file",
                    flags = ["-E"],
                ),
                flag_group(
                    expand_if_available = "dependency_file",
                    flags = ["-MD", "-MF", "%{dependency_file}"],
                ),
                flag_group(
                    expand_if_available = "output_file",
                    flags = ["-frandom-seed=%{output_file}"],
                ),
            ],
        ),
        flag_set(
            # Flags specific to compiling C++ sources.
            actions = all_cpp_compile_actions,
            flag_groups = [flag_group(flags = [
                "-std=c++20",
            ])],
        ),
        flag_set(
            actions = codegen_compile_actions,
            flag_groups = [flag_group(flags = [
                "-ffunction-sections",
                "-fdata-sections",
            ])],
        ),
        flag_set(
            actions = codegen_compile_actions,
            flag_groups = [flag_group(
                expand_if_available = "pic",
                flags = ["-fPIC"],
            )],
        ),
        flag_set(
            actions = preprocessor_compile_actions,
            flag_groups = [
                flag_group(flags = [
                    # Disable a warning and override builtin macros to
                    # ensure a hermetic build.
                    "-Wno-builtin-macro-redefined",
                    "-D__DATE__=\"redacted\"",
                    "-D__TIMESTAMP__=\"redacted\"",
                    "-D__TIME__=\"redacted\"",
                ]),
                flag_group(
                    flags = ["-D%{preprocessor_defines}"],
                    iterate_over = "preprocessor_defines",
                ),
                flag_group(
                    expand_if_available = "includes",
                    flags = ["-include", "%{includes}"],
                    iterate_over = "includes",
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
                    expand_if_available = "strip_debug_symbols",
                    flags = ["-Wl,-S"],
                ),
                flag_group(
                    expand_if_available = "library_search_directories",
                    flags = ["-L%{library_search_directories}"],
                    iterate_over = "library_search_directories",
                ),
                flag_group(
                    expand_if_available =
                        "runtime_library_search_directories",
                    iterate_over = "runtime_library_search_directories",
                    flags = [
                        "-Wl,-rpath,$ORIGIN/%{runtime_library_search_directories}",
                    ],
                ),
            ],
        ),
        flag_set(
            actions = all_link_actions,
            flag_groups = [
                flag_group(
                    flags = [
                        "-fuse-ld=lld",
                        # Force the C++ standard library and runtime libraries
                        # to be statically linked. This works even with libc++
                        # and libunwind despite the names, provided libc++ is
                        # built with the CMake option:
                        # - `-DCMAKE_POSITION_INDEPENDENT_CODE=ON`
                        "-static-libstdc++",
                        "-static-libgcc",
                        # Link with Clang's runtime library. This is always
                        # linked statically.
                        "-rtlib=compiler-rt",
                        # Link with pthread.
                        "-lpthread",
                    ],
                ),
            ],
            with_features = [with_feature_set(["linux_target"])],
        ),
        flag_set(
            actions = [ACTION_NAMES.cpp_link_executable],
            flag_groups = [flag_group(
                expand_if_available = "force_pic",
                flags = ["-pie"],
            )],
            with_features = [with_feature_set([
                "linux_target",
                "freebsd_target",
            ])],
        ),
        flag_set(
            actions = [ACTION_NAMES.cpp_link_executable],
            flag_groups = [flag_group(
                expand_if_available = "force_pic",
                flags = ["-fpie"],
            )],
            with_features = [with_feature_set(["macos_target"])],
        ),
    ],
)

clang_warnings_feature = feature(
    name = "clang_warnings",
    enabled = True,
    flag_sets = [flag_set(
        actions = all_compile_actions,
        flag_groups = [flag_group(flags = [
            "-Werror",
            "-Wall",
            "-Wextra",
            "-Wthread-safety",
            "-Wself-assign",
            "-Wimplicit-fallthrough",
            "-Wctad-maybe-unsupported",
            "-Wextra-semi",
            "-Wmissing-prototypes",
            "-Wzero-as-null-pointer-constant",
            "-Wdelete-non-virtual-dtor",

            # TODO: Regression that warns on anonymous unions; remove depending
            # on fix.
            "-Wno-missing-designated-field-initializers",
        ])],
    )],
)

# Libc++ HARDENING_MODE has 4 possible values:
# https://libcxx.llvm.org/Hardening.html#notes-for-users
#
# Do not enable DEBUG hardening mode, even for -c dbg, because its performance
# impact on llvm-symbolizer is too severe -- this flag results in symbolization
# becoming quadratic in the number of debug symbols, in practice meaning it
# never completes.
_libcpp_debug_flags = [
    "-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE",
]
_libcpp_release_flags = [
    "-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST",
]

def libcxx_feature(llvm_bindir = None, clang_bindir = None):
    """Builds a libc++ feature.

    Returns:
        The feature for use with `cc_toolchain_config`.

    Args:
        llvm_bindir: Optional LLVM installation `bin` directory, causes the
            feature to look for adjacent installed libraries.
        clang_bindir: Optional Clang installation `bin` directory, causes the
            feature to look for adjacent installed libraries if different from
            `llvm_bindir`.
    """

    # Explicitly add LLVM libs to the search path to preempt the
    # detected GCC installation's library paths. Those might have a
    # system installed libc++ and we want to find the one next to
    # our Clang.
    extra_link_flags = []
    if llvm_bindir:
        extra_link_flags.append("-L" + llvm_bindir + "/../lib")
    if clang_bindir and clang_bindir != llvm_bindir:
        extra_link_flags.append("-L" + clang_bindir + "/../lib")

    return feature(
        name = "libcxx",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_cpp_compile_actions + all_link_actions,
                flag_groups = [flag_group(flags = [
                    "-stdlib=libc++",
                ])],
                with_features = [
                    # libc++ is only used on non-Windows platforms.
                    with_feature_set(not_features = ["windows_target"]),
                ],
            ),
            flag_set(
                actions = all_cpp_compile_actions,
                flag_groups = [flag_group(flags = _libcpp_debug_flags)],
                with_features = [with_feature_set(not_features = ["opt"])],
            ),
            flag_set(
                actions = all_cpp_compile_actions,
                flag_groups = [flag_group(flags = _libcpp_release_flags)],
                with_features = [with_feature_set(features = ["opt"])],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = [
                    "-unwindlib=libunwind",
                ])],
                with_features = [
                    # libc++ is only used on non-Windows platforms.
                    with_feature_set(not_features = ["windows_target"]),
                ],
            ),
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = extra_link_flags + [
                    # Force linking the static libc++abi archive here. This
                    # *should* be linked automatically, but not every release of
                    # LLVM correctly sets the CMake flags to do so.
                    "-l:libc++abi.a",
                ])],
                with_features = [with_feature_set(["linux_target"])],
            ),
        ],
    )
