; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep extractvalue

; Instcombine should fold various combinations of insertvalue and extractvalue
; together
declare void @bar({i32, i32} %a)

define i32 @foo() {
        ; Build a simple struct and pull values out again
        %s1.1 = insertvalue {i32, i32} undef, i32 0, 0
        %s1 = insertvalue {i32, i32} %s1.1, i32 1, 1
        %v1 = extractvalue {i32, i32} %s1, 0
        %v2 = extractvalue {i32, i32} %s1, 1
        
        ; Build a nested struct and pull a sub struct out of it
        ; This requires instcombine to insert a few insertvalue instructions
        %ns1.1 = insertvalue {i32, {i32, i32}} undef, i32 %v1, 0
        %ns1.2 = insertvalue {i32, {i32, i32}} %ns1.1, i32 %v1, 1, 0
        %ns1   = insertvalue {i32, {i32, i32}} %ns1.2, i32 %v2, 1, 1
        %s2    = extractvalue {i32, {i32, i32}} %ns1, 1
        call void @bar({i32, i32} %s2)
        %v3 = extractvalue {i32, {i32, i32}} %ns1, 1, 1
        ret i32 %v3
}

