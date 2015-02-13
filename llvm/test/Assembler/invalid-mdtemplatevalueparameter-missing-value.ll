; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: [[@LINE+2]]:53: error: missing required field 'value'
!0 = !MDTemplateValueParameter(tag: DW_TAG_template_value_parameter,
                               scope: !{}, type: !{})
