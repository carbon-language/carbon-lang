# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Install-related rules."""

def _install_path_rule_impl(ctx):
    """Turns the install path into a variable for rules to use."""
    return [
        platform_common.TemplateVariableInfo({
            "INSTALL_PATH": ctx.build_setting_value,
        }),
    ]

install_path_rule = rule(
    implementation = _install_path_rule_impl,
    build_setting = config.string(flag = True),
)
