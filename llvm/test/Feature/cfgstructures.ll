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

;; This is a pair of loops that share the same header

void "sharedheader"(bool %cond)
begin
	br label %A
A:
	br bool %cond, label %X, label %Y

X:
	br label %A
Y:
	br label %A
end

;; This is a simple nested loop
void "nested"(bool %cond1, bool %cond2, bool %cond3)
begin
	br label %Loop1

Loop1:
	br label %Loop2

Loop2:
	br label %Loop3

Loop3:
	br bool %cond3, label %Loop3, label %L3Exit

L3Exit:
	br bool %cond2, label %Loop2, label %L2Exit

L2Exit:
	br bool %cond1, label %Loop1, label %L1Exit

L1Exit:
	ret void
end

