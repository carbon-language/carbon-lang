; This testcase is not level raised properly...
;
; RUN: as < %s | opt -raise | dis | grep ' cast ' | not grep '*'

	%List = type { int, %List* }

implementation

%List* "createList"(uint %Depth)
begin
	%reg110 = malloc uint, uint 4
	store uint %Depth, uint* %reg110
	%reg113 = call %List* %createList( uint %Depth )
	%reg217 = getelementptr uint* %reg110, long 2
	%cast221 = cast uint* %reg217 to %List**
	store %List* %reg113, %List** %cast221
	%cast222 = cast uint* %reg110 to %List*
	ret %List* %cast222
end

