; Crash in post dominator set construction.
;
; RUN: analyze -postdomset %s
;

implementation

int "looptest"()
begin
        br label %L2Top

L2Top:
	br bool true, label %L2End, label %L2Top

L2Body:
	br label %L2Top

L2End:
	ret int 0
end

