
int %main() {
	%X = add double 0.0, 1.0
	%Y = sub double 0.0, 1.0
	%Z = seteq double %X, %Y
	add double %Y, 0.0
	ret int 0
}
