; This should parse correctly without an 'implementation', but there seems to 
; be a problem...

	%List = type { int, %List* }

%List* "test"()
begin
	ret %List* null
end

