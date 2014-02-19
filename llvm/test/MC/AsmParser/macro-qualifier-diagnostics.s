# RUN: not llvm-mc -triple i386 -o /dev/null %s 2>&1 | FileCheck %s

	.macro missing_qualifier parameter:

# CHECK: error: missing parameter qualifier for 'parameter' in macro 'missing_qualifier'
# CHECK: 	.macro missing_qualifier parameter:
# CHECK:                                           ^

	.macro non_identifier_qualifier parameter:0

# CHECK: error: missing parameter qualifier for 'parameter' in macro 'non_identifier_qualifier'
# CHECK: 	.macro non_identifier_qualifier parameter:0
# CHECK:                                                  ^

	.macro invalid_qualifier parameter:invalid_qualifier

# CHECK: error: invalid_qualifier is not a valid parameter qualifier for 'parameter' in macro 'invalid_qualifier'
# CHECK: 	.macro invalid_qualifier parameter:invalid_qualifier
# CHECK:                                           ^

	.macro pointless_default parameter:req=default
	.endm

# CHECK: warning: pointless default value for required parameter 'parameter' in macro 'pointless_default'
# CHECK: 	.macro pointless_default parameter:req=default
# CHECK:                                               ^

	.macro missing_required_parameter parameter:req
	.endm

	missing_required_parameter

# CHECK: error: missing value for required parameter 'parameter' in macro 'missing_required_parameter'
# CHECK: 	missing_required_parameter
# CHECK:                                  ^

	.macro missing_second_required_argument first=0 second:req
	.endm

	missing_second_required_argument

# CHECK: error: missing value for required parameter 'second' in macro 'missing_second_required_argument'
# CHECK: 	missing_second_required_argument
# CHECK:                                        ^

	.macro second_third_required first=0 second:req third:req
	.endm

	second_third_required 0

# CHECK: error: missing value for required parameter 'second' in macro 'second_third_required'
# CHECK: 	second_third_required 0
# CHECK:                               ^

# CHECK: error: missing value for required parameter 'third' in macro 'second_third_required'
# CHECK: 	second_third_required 0
# CHECK:                               ^

	second_third_required third=3 first=1

# CHECK: error: missing value for required parameter 'second' in macro 'second_third_required'
# CHECK: 	second_third_required third=3 first=1
# CHECK:                                             ^

