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
    "preprocessor_compile_actions",
)
load(
    ":cc_toolchain_base_features.bzl",
    "base_features",
    "output_flags_feature",
    "user_flags_feature",
)
load(
    ":cc_toolchain_config_features.bzl",
    "target_cpu_features",
    "target_os_features",
)
load(
    ":cc_toolchain_cpp_features.bzl",
    "clang_feature",
    "clang_warnings_feature",
    "libcxx_feature",
)
load(":cc_toolchain_debugging.bzl", "debugging_features")
load(":cc_toolchain_linking.bzl", "linking_features")
load(":cc_toolchain_modules.bzl", "modules_features")
load(":cc_toolchain_optimization.bzl", "optimization_features")
load(":cc_toolchain_sanitizer_features.bzl", "sanitizer_features")
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
    project_flags_feature = feature(
        name = "project_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_compile_actions,
                flag_groups = [flag_group(flags = [
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
                ])],
            ),
            flag_set(
                actions = preprocessor_compile_actions,
                flag_groups = [flag_group(flags = [
                    # Pass the Clang version to get better Bazel caching
                    # behavior with system-installed Clang binaries.
                    "-DCLANG_VERSION_FOR_CACHE=\"%s\"" % clang_version_for_cache,
                ])],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [flag_group(flags = ["-DHAVE_MALLCTL"])],
                with_features = [with_feature_set(["freebsd_target"])],
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

    # The order of the features determines the relative order of flags used.
    features = []
    features += base_features
    features += target_os_features(ctx.attr.target_os)
    features += target_cpu_features(ctx.attr.target_cpu)
    features += [
        # We always use Clang in the toolchain and enable all of its warnings.
        clang_feature,
        clang_warnings_feature,
        # Enable libc++ where supported.
        libcxx_feature(llvm_bindir, clang_bindir),

        # Include any project-specific flags, importantly after the Clang and
        # warnings features so we can override as necessary here.
        project_flags_feature,
    ]
    features += sanitizer_features
    features += optimization_features
    features += modules_features
    features += debugging_features
    features += linking_features

    # TODO: Refactor the target-specific feature management here to be part of
    # building `linking_features`.
    features += [
        feature(name = "supports_dynamic_linker", enabled = ctx.attr.target_os == "linux"),
        feature(name = "supports_start_end_lib", enabled = ctx.attr.target_os == "linux"),
    ]

    # Lastly, we add a feature that enables others in the default `fastbuild`
    # mode. This is also a good place to add any project-specific features.
    features.append(enable_in_fastbuild)

    # Add user flags and the output flags at the end.
    features.append(user_flags_feature)
    features.append(output_flags_feature)
    return features

def _impl(ctx):
    # Only use a sysroot if one was found when detecting Clang.
    sysroot = None
    if sysroot_dir != "None":
        sysroot = sysroot_dir

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
