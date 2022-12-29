/*
 * Part of the Carbon Language project, under the Apache License v2.0 with LLVM
 * Exceptions. See /LICENSE for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

/**
 * @fileoverview Description of this file.
 */
module.exports = grammar({
  name: 'carbon',

  source_file: ($) => $.package,

  extras: ($) => [/\s/, $.comment],

  rules: {
    // TODO: add the actual grammar rules
    source_file: ($) => seq($.package_directive, $.declaration_list),

    package_directive: ($) =>
      seq('package', $.identifier, $.package_api_or_impl, ';'),

    declaration_list: ($) => 'todo',

    package_api_or_impl: ($) => choice('api', 'impl'),

    comment: ($) => token(seq('//', /.*/)),

    identifier: ($) => /[A-Za-z_][A-Za-z0-9_]*/,
  },
});
