
int %main() {
	%double1 = add double 0.0, 0.0
	%double2 = add double 0.0, 0.0
	%float1 = add float 0.0, 0.0
	%float2 = add float 0.0, 0.0
	%test49 = seteq float %float1, %float2
	%test50 = setge float %float1, %float2
	%test51 = setgt float %float1, %float2
	%test52 = setle float %float1, %float2
	%test53 = setlt float %float1, %float2
	%test54 = setne float %float1, %float2
	%test55 = seteq double %double1, %double2
	%test56 = setge double %double1, %double2
	%test57 = setgt double %double1, %double2
	%test58 = setle double %double1, %double2
	%test59 = setlt double %double1, %double2
	%test60 = setne double %double1, %double2
	ret int 0
}
