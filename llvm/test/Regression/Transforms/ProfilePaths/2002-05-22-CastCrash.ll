; RUN: llvm-as < %s | opt -paths
;
%globalCrc = uninitialized global uint         ; <uint*> [#uses=1]

void "initialiseCRC"() {
bb1:                                    ;[#uses=0]
        store uint 4294967295, uint* %globalCrc
        ret void
}

