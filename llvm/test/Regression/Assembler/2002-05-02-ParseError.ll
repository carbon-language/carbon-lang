; This should parse correctly without an 'implementation', but there seems to 
; be a problem...

	%T = type int *

%T "test"()
begin
	ret %T null
end

