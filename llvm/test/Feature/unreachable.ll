
implementation

declare void %bar()

int %foo() {  ;; Calling this function has undefined behavior
	unreachable
}

double %xyz() {
	call void %bar()
	unreachable          ;; Bar must not return.
}
