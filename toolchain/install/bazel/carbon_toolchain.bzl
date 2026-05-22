# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Module extension to configure Carbon's `cc_toolchain`s.

This extension extracts configuration from the Carbon toolchain into
`carbon_detected_variables.bzl`. These values are then used by the
`cc_toolchain` to setup the Carbon toolchain as a viable C++ Bazel toolchain.
"""

def _compute_config_vars(repository_ctx, carbon):
    """Runs the `carbon` binary to get its config variables."""
    exec_result = repository_ctx.execute([carbon, "config", "--json"])
    if exec_result.return_code != 0:
        fail("Command failed with return code {0}:\n{1}".format(
            exec_result.return_code,
            exec_result.stderr,
        ))

    vars = json.decode(exec_result.stdout)
    if type(vars) != "dict":
        fail("Config JSON decoded to a non-dict value: \n" + exec_result.stdout)

    # Turn the values of all the keys in the JSON config results into strings.
    # This provides a dictionary suitable for substituting with
    # `repository_ctx.template`.
    return {key: str(value) for key, value in vars.items()}

def _create_config_repo_impl(repository_ctx):
    vars = _compute_config_vars(repository_ctx, repository_ctx.attr._carbon)

    repository_ctx.template(
        "carbon_detected_variables.bzl",
        repository_ctx.attr._template,
        vars,
    )

    repository_ctx.file("BUILD.bazel", """
exports_files(["carbon_detected_variables.bzl"])
""")

_create_config_repo = repository_rule(
    implementation = _create_config_repo_impl,
    attrs = {
        "_carbon": attr.label(
            default = "//:carbon-busybox",
            allow_single_file = True,
        ),
        "_template": attr.label(
            default = "//bazel:carbon_detected_variables.tpl.bzl",
            allow_single_file = True,
        ),
    },
)

carbon_toolchain_config = module_extension(
    implementation =
        lambda ctx: _create_config_repo(name = "carbon_toolchain_config"),
)
