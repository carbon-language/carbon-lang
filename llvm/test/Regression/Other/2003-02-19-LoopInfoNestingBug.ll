; LoopInfo is incorrectly calculating loop nesting!  In this case it doesn't 
; figure out that loop "Inner" should be nested inside of leep "LoopHeader", 
; and instead nests it just inside loop "Top"
;
; RUN: analyze -loops %s | grep '     Loop Containing:[ ]*%Inner'
;

implementation

void %test() {
	br label %Top
Top:
	br label %LoopHeader
Next:
	br bool false, label %Inner, label %Out
Inner:
	br bool false, label %Inner, label %LoopHeader

LoopHeader:
	br label %Next

Out:
	br bool false, label %Top, label %Done

Done:
	ret void
}
