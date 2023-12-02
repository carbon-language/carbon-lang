// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: - filename: keywords.carbon
// CHECK:STDOUT:   tokens: [
// CHECK:STDOUT:     { index:  0, kind:           'FileStart', line: {{ *\d+}}, column:  1, indent: 1, spelling: '', has_trailing_space: true },

abstract
// CHECK:STDOUT:     { index:  1, kind:            'Abstract', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'abstract', has_trailing_space: true },
adapt
// CHECK:STDOUT:     { index:  2, kind:               'Adapt', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'adapt', has_trailing_space: true },
addr
// CHECK:STDOUT:     { index:  3, kind:                'Addr', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'addr', has_trailing_space: true },
alias
// CHECK:STDOUT:     { index:  4, kind:               'Alias', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'alias', has_trailing_space: true },
and
// CHECK:STDOUT:     { index:  5, kind:                 'And', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'and', has_trailing_space: true },
api
// CHECK:STDOUT:     { index:  6, kind:                 'Api', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'api', has_trailing_space: true },
as
// CHECK:STDOUT:     { index:  7, kind:                  'As', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'as', has_trailing_space: true },
auto
// CHECK:STDOUT:     { index:  8, kind:                'Auto', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'auto', has_trailing_space: true },
base
// CHECK:STDOUT:     { index:  9, kind:                'Base', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'base', has_trailing_space: true },
bool
// CHECK:STDOUT:     { index: 10, kind:                'Bool', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'bool', has_trailing_space: true },
break
// CHECK:STDOUT:     { index: 11, kind:               'Break', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'break', has_trailing_space: true },
case
// CHECK:STDOUT:     { index: 12, kind:                'Case', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'case', has_trailing_space: true },
choice
// CHECK:STDOUT:     { index: 13, kind:              'Choice', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'choice', has_trailing_space: true },
class
// CHECK:STDOUT:     { index: 14, kind:               'Class', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'class', has_trailing_space: true },
const
// CHECK:STDOUT:     { index: 15, kind:               'Const', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'const', has_trailing_space: true },
constraint
// CHECK:STDOUT:     { index: 16, kind:          'Constraint', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'constraint', has_trailing_space: true },
continue
// CHECK:STDOUT:     { index: 17, kind:            'Continue', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'continue', has_trailing_space: true },
default
// CHECK:STDOUT:     { index: 18, kind:             'Default', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'default', has_trailing_space: true },
destructor
// CHECK:STDOUT:     { index: 19, kind:          'Destructor', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'destructor', has_trailing_space: true },
else
// CHECK:STDOUT:     { index: 20, kind:                'Else', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'else', has_trailing_space: true },
extend
// CHECK:STDOUT:     { index: 21, kind:              'Extend', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'extend', has_trailing_space: true },
false
// CHECK:STDOUT:     { index: 22, kind:               'False', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'false', has_trailing_space: true },
final
// CHECK:STDOUT:     { index: 23, kind:               'Final', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'final', has_trailing_space: true },
fn
// CHECK:STDOUT:     { index: 24, kind:                  'Fn', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'fn', has_trailing_space: true },
for
// CHECK:STDOUT:     { index: 25, kind:                 'For', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'for', has_trailing_space: true },
forall
// CHECK:STDOUT:     { index: 26, kind:              'Forall', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'forall', has_trailing_space: true },
friend
// CHECK:STDOUT:     { index: 27, kind:              'Friend', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'friend', has_trailing_space: true },
if
// CHECK:STDOUT:     { index: 28, kind:                  'If', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'if', has_trailing_space: true },
impl
// CHECK:STDOUT:     { index: 29, kind:                'Impl', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'impl', has_trailing_space: true },
impls
// CHECK:STDOUT:     { index: 30, kind:               'Impls', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'impls', has_trailing_space: true },
import
// CHECK:STDOUT:     { index: 31, kind:              'Import', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'import', has_trailing_space: true },
in
// CHECK:STDOUT:     { index: 32, kind:                  'In', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'in', has_trailing_space: true },
interface
// CHECK:STDOUT:     { index: 33, kind:           'Interface', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'interface', has_trailing_space: true },
let
// CHECK:STDOUT:     { index: 34, kind:                 'Let', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'let', has_trailing_space: true },
library
// CHECK:STDOUT:     { index: 35, kind:             'Library', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'library', has_trailing_space: true },
like
// CHECK:STDOUT:     { index: 36, kind:                'Like', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'like', has_trailing_space: true },
match
// CHECK:STDOUT:     { index: 37, kind:               'Match', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'match', has_trailing_space: true },
namespace
// CHECK:STDOUT:     { index: 38, kind:           'Namespace', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'namespace', has_trailing_space: true },
not
// CHECK:STDOUT:     { index: 39, kind:                 'Not', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'not', has_trailing_space: true },
observe
// CHECK:STDOUT:     { index: 40, kind:             'Observe', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'observe', has_trailing_space: true },
or
// CHECK:STDOUT:     { index: 41, kind:                  'Or', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'or', has_trailing_space: true },
package
// CHECK:STDOUT:     { index: 42, kind:             'Package', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'package', has_trailing_space: true },
partial
// CHECK:STDOUT:     { index: 43, kind:             'Partial', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'partial', has_trailing_space: true },
private
// CHECK:STDOUT:     { index: 44, kind:             'Private', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'private', has_trailing_space: true },
protected
// CHECK:STDOUT:     { index: 45, kind:           'Protected', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'protected', has_trailing_space: true },
require
// CHECK:STDOUT:     { index: 46, kind:             'Require', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'require', has_trailing_space: true },
return
// CHECK:STDOUT:     { index: 47, kind:              'Return', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'return', has_trailing_space: true },
returned
// CHECK:STDOUT:     { index: 48, kind:            'Returned', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'returned', has_trailing_space: true },
Self
// CHECK:STDOUT:     { index: 49, kind:  'SelfTypeIdentifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'Self', has_trailing_space: true },
self
// CHECK:STDOUT:     { index: 50, kind: 'SelfValueIdentifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'self', has_trailing_space: true },
String
// CHECK:STDOUT:     { index: 51, kind:   'StringTypeLiteral', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'String', has_trailing_space: true },
template
// CHECK:STDOUT:     { index: 52, kind:            'Template', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'template', has_trailing_space: true },
then
// CHECK:STDOUT:     { index: 53, kind:                'Then', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'then', has_trailing_space: true },
true
// CHECK:STDOUT:     { index: 54, kind:                'True', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'true', has_trailing_space: true },
type
// CHECK:STDOUT:     { index: 55, kind:                'Type', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'type', has_trailing_space: true },
_
// CHECK:STDOUT:     { index: 56, kind:          'Underscore', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: '_', has_trailing_space: true },
var
// CHECK:STDOUT:     { index: 57, kind:                 'Var', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'var', has_trailing_space: true },
virtual
// CHECK:STDOUT:     { index: 58, kind:             'Virtual', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'virtual', has_trailing_space: true },
where
// CHECK:STDOUT:     { index: 59, kind:               'Where', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'where', has_trailing_space: true },
while
// CHECK:STDOUT:     { index: 60, kind:               'While', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'while', has_trailing_space: true },

notakeyword
// CHECK:STDOUT:     { index: 61, kind:          'Identifier', line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: 'notakeyword', identifier: 0, has_trailing_space: true },

// CHECK:STDOUT:     { index: 62, kind:             'FileEnd', line: {{ *}}[[@LINE+1]], column: {{ *\d+}}, indent: 1, spelling: '' },
// CHECK:STDOUT:   ]
