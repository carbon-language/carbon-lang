implementation

;; This is an irreducible flow graph

void "irreducible"(bool %cond)
begin
	br bool %cond, label %X, label %Y

X:
	br label %Y
Y:
	br label %X
end

