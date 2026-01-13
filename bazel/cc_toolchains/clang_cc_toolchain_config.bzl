# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A Starlark cc_toolchain configuration rule"""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load(
    "@rules_cc//cc:cc_toolchain_config_lib.bzl",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "with_feature_set",
)
load("@rules_cc//cc:defs.bzl", "cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load(
    ":cc_toolchain_actions.bzl",
    "all_compile_actions",
    "all_cpp_compile_actions",
    "all_link_actions",
    "codegen_compile_actions",
    "preprocessor_compile_actions",
)
load(
    ":cc_toolchain_base_features.bzl",
    "base_features",
    "output_flags_feature",
    "user_flags_feature",
)
load(":cc_toolchain_debugging.bzl", "debugging_features")
load(
    ":cc_toolchain_linking.bzl",
    "default_link_libraries_feature",
    "linking_features",
    "macos_link_libraries_feature",
)
load(":cc_toolchain_modules.bzl", "modules_features")
load(
    ":cc_toolchain_optimization.bzl",
    "aarch64_cpu_flags",
    "optimization_features",
    "x86_64_cpu_flags",
)
load(
    ":cc_toolchain_sanitizer_features.bzl",
    "macos_asan_workarounds",
    "sanitizer_features",
    "sanitizer_static_lib_flags",
)
load(
    ":cc_toolchain_tools.bzl",
    "llvm_action_configs",
    "llvm_tool_paths",
)
load(
    ":clang_detected_variables.bzl",
    "clang_bindir",
    "clang_include_dirs_list",
    "clang_resource_dir",
    "clang_version_for_cache",
    "llvm_bindir",
    "sysroot_dir",
)

def _build_features(ctx):
    std_compile_flags = ["-std=c++20"]

    # libc++ is only used on non-Windows platforms.
    if ctx.attr.target_os != "windows":
        std_compile_flags.append("-stdlib=libc++")

    # TODO: Refactor this into a reusable form in its own file.
    default_flags_feature = feature(
        name = "default_flags",
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

                        # TODO: Regression that warns on anonymous unions;
                        # remove depending on fix.
                        "-Wno-missing-designated-field-initializers",

                        # Don't warn on external code as we can't
                        # necessarily patch it easily. Note that these have
                        # to be initial directories in the `#include` line.
                        "--system-header-prefix=absl/",
                        "--system-header-prefix=benchmark/",
                        "--system-header-prefix=boost/",
                        "--system-header-prefix=clang-tools-extra/",
                        "--system-header-prefix=clang/",
                        "--system-header-prefix=gmock/",
                        "--system-header-prefix=gtest/",
                        "--system-header-prefix=libfuzzer/",
                        "--system-header-prefix=llvm/",
                        "--system-header-prefix=re2/",
                        "--system-header-prefix=tools/cpp/",
                        "--system-header-prefix=tree_sitter/",
                        # Compile actions shouldn't link anything.
                        "-c",
                    ]),
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
                actions = all_cpp_compile_actions + all_link_actions,
                flag_groups = [flag_group(flags = std_compile_flags)],
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
                        # Pass the clang version as a define so that bazel
                        # caching is more likely to notice version changes.
                        "-DCLANG_VERSION_FOR_CACHE=\"%s\"" % clang_version_for_cache,

                        # Enable the use of zlib and zstd in LLVM. We define
                        # these here and use the normal Bazel builds of both
                        # rather than using the custom zlib and zstd `BUILD`
                        # files shipped with LLVM that use `defines`.
                        "-DLLVM_ENABLE_ZLIB",
                        "-DLLVM_ENABLE_ZSTD",
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
        ],
    )

    # An enabled feature that requires the `fastbuild` compilation. This is used
    # to toggle general features on by default, while allowing them to be
    # directly enabled and disabled more generally as desired.
    enable_in_fastbuild = feature(
        name = "enable_in_fastbuild",
        enabled = True,
        requires = [feature_set(["fastbuild"])],
        implies = [
            "asan",
            "asan_min_size",
            "minimal_optimization_flags",
            "minimal_debug_info_flags",
        ],
    )

    # Clang HARDENING_MODE has 4 possible values:
    # https://libcxx.llvm.org/Hardening.html#notes-for-users
    #
    # Do not enable DEBUG hardening mode, even for -c dbg, because its
    # performance impact on llvm-symbolizer is too severe -- this flag
    # results in symbolization becoming quadratic in the number of debug
    # symbols, in practice meaning it never completes.
    libcpp_debug_flags = [
        "-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE",
    ]
    libcpp_release_flags = [
        "-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST",
    ]

    # TODO: Refactor this into a reusable form in its own file.
    linux_flags_feature = feature(
        name = "linux_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(
                    flags = [
                        "-fuse-ld=lld",
                        "-stdlib=libc++",
                        "-unwindlib=libunwind",
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
                        # Explicitly add LLVM libs to the search path to preempt
                        # the detected GCC installation's library paths. Those
                        # might have a system installed libc++ and we want to
                        # find the one next to our Clang.
                        "-L" + llvm_bindir + "/../lib",
                        # Link with pthread.
                        "-lpthread",
                        # Force linking the static libc++abi archive here. This
                        # *should* be linked automatically, but not every
                        # release of LLVM correctly sets the CMake flags to do
                        # so.
                        "-l:libc++abi.a",
                    ],
                )],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = [flag_group(flags = libcpp_debug_flags)],
                with_features = [with_feature_set(not_features = ["opt"])],
            ),
            flag_set(
                actions = all_compile_actions,
                flag_groups = [flag_group(flags = libcpp_release_flags)],
                with_features = [with_feature_set(features = ["opt"])],
            ),
            flag_set(
                actions = [ACTION_NAMES.cpp_link_executable],
                flag_groups = [flag_group(
                    expand_if_available = "force_pic",
                    flags = ["-pie"],
                )],
            ),
        ],
    )

    # TODO: Refactor this into a reusable form in its own file.
    macos_flags_feature = feature(
        name = "macos_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_link_executable],
                flag_groups = [flag_group(
                    expand_if_available = "force_pic",
                    flags = ["-fpie"],
                )],
            ),
        ],
    )

    # TODO: Refactor this into a reusable form in its own file.
    freebsd_flags_feature = feature(
        name = "freebsd_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [flag_group(flags = ["-DHAVE_MALLCTL"])],
            ),
            flag_set(
                actions = [ACTION_NAMES.cpp_link_executable],
                flag_groups = [flag_group(
                    expand_if_available = "force_pic",
                    flags = ["-pie"],
                )],
            ),
        ],
    )

    # The order of the features determines the relative order of flags used.
    features = []
    features += base_features
    features.append(default_flags_feature)
    features += sanitizer_features

    features += optimization_features

    # TODO: Refactor target-specific feature management to be part of
    # `optimization_features`.
    if ctx.attr.target_cpu in ["aarch64", "arm64"]:
        features.append(aarch64_cpu_flags)
    else:
        features.append(x86_64_cpu_flags)

    features += modules_features
    features += debugging_features

    # Next, add the features based on the target platform. Here too the
    # features are order sensitive.
    if ctx.attr.target_os == "linux":
        features.append(sanitizer_static_lib_flags)
        features.append(linux_flags_feature)
        sysroot = None
    elif ctx.attr.target_os == "windows":
        # TODO: Need to figure out if we need to add windows specific features
        # I think the .pdb debug files will need to be handled differently,
        # so that might be an example where a feature must be added.
        sysroot = None
    elif ctx.attr.target_os == "macos":
        features.append(macos_asan_workarounds)
        features.append(macos_flags_feature)
        sysroot = sysroot_dir
    elif ctx.attr.target_os == "freebsd":
        features.append(sanitizer_static_lib_flags)
        features.append(freebsd_flags_feature)
        sysroot = sysroot_dir
    else:
        fail("Unsupported target OS!")

    # Next, append the libraries to link.
    features += linking_features

    # TODO: Refactor the target-specific feature management here to be part of
    # building `linking_features`.
    features += [
        feature(name = "supports_dynamic_linker", enabled = ctx.attr.target_os == "linux"),
        feature(name = "supports_start_end_lib", enabled = ctx.attr.target_os == "linux"),
    ]
    if ctx.attr.target_os == "macos":
        features.append(macos_link_libraries_feature)
    else:
        features.append(default_link_libraries_feature)

    # Lastly, we add a feature that enables others in the default `fastbuild`
    # mode. This is also a good place to add any project-specific features.
    features.append(enable_in_fastbuild)

    # Add user flags and the output flags at the end.
    features.append(user_flags_feature)
    features.append(output_flags_feature)
    return features

def _impl(ctx):
    # TODO: See if this can be refactored into platform features.
    if ctx.attr.target_os == "linux":
        sysroot = None
    elif ctx.attr.target_os == "windows":
        sysroot = None
    elif ctx.attr.target_os == "macos":
        sysroot = sysroot_dir
    elif ctx.attr.target_os == "freebsd":
        sysroot = sysroot_dir
    else:
        fail("Unsupported target OS!")

    identifier = "local-{0}-{1}".format(ctx.attr.target_cpu, ctx.attr.target_os)
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = _build_features(ctx),
        action_configs = llvm_action_configs(llvm_bindir, clang_bindir),
        cxx_builtin_include_directories = clang_include_dirs_list + [
            # Add Clang's resource directory to the end of the builtin include
            # directories to cover the use of sanitizer resource files by the
            # driver.
            clang_resource_dir + "/share",
        ],
        builtin_sysroot = sysroot,

        # This configuration only supports local non-cross builds so derive
        # everything from the target CPU selected.
        toolchain_identifier = identifier,
        host_system_name = identifier,
        target_system_name = identifier,
        target_cpu = ctx.attr.target_cpu,

        # This is used to expose a "flag" that `config_setting` rules can use to
        # determine if the compiler is Clang.
        compiler = "clang",

        # These attributes aren't meaningful at all so just use placeholder
        # values.
        target_libc = "local",
        abi_version = "local",
        abi_libc_version = "local",

        # We do have to pass in our tool paths.
        tool_paths = llvm_tool_paths(llvm_bindir, clang_bindir),
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
        "target_cpu": attr.string(mandatory = True),
        "target_os": attr.string(mandatory = True),
    },
    provides = [CcToolchainConfigInfo],
)

def cc_local_toolchain_suite(name, configs):
    """Create a toolchain suite that uses the local Clang/LLVM install.

    Args:
        name: The name of the toolchain suite to produce.
        configs: An array of (os, cpu) pairs to support in the toolchain.
    """

    # An empty filegroup to use when stubbing out the toolchains.
    native.filegroup(
        name = name + "_empty",
        srcs = [],
    )

    # Create the individual local toolchains for each CPU.
    for (os, cpu) in configs:
        config_name = "{0}_{1}_{2}".format(name, os, cpu)
        cc_toolchain_config(
            name = config_name + "_config",
            target_os = os,
            target_cpu = cpu,
        )
        cc_toolchain(
            name = config_name + "_tools",
            all_files = ":" + name + "_empty",
            ar_files = ":" + name + "_empty",
            as_files = ":" + name + "_empty",
            compiler_files = ":" + name + "_empty",
            dwp_files = ":" + name + "_empty",
            linker_files = ":" + name + "_empty",
            objcopy_files = ":" + name + "_empty",
            strip_files = ":" + name + "_empty",
            supports_param_files = 1,
            toolchain_config = ":" + config_name + "_config",
            toolchain_identifier = config_name,
        )
        compatible_with = ["@platforms//cpu:" + cpu, "@platforms//os:" + os]
        native.toolchain(
            name = config_name,
            exec_compatible_with = compatible_with,
            target_compatible_with = compatible_with,
            toolchain = config_name + "_tools",
            toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
        )
