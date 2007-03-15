; RUN: llvm-as < %s | llc -march=x86 -mattr=+mmx

;; A basic sanity check to make sure that MMX arithmetic actually compiles.

define void @foo(<8 x i8>* %A, <8 x i8>* %B) {
entry:
	%tmp5 = load <8 x i8>* %A		; <<8 x i8>> [#uses=1]
	%tmp7 = load <8 x i8>* %B		; <<8 x i8>> [#uses=1]
	%tmp8 = add <8 x i8> %tmp5, %tmp7		; <<8 x i8>> [#uses=2]
	store <8 x i8> %tmp8, <8 x i8>* %A
	%tmp14 = load <8 x i8>* %B		; <<8 x i8>> [#uses=1]
	%tmp25 = tail call <8 x i8> @llvm.x86.mmx.padds.b( <8 x i8> %tmp14, <8 x i8> %tmp8 )		; <<8 x i8>> [#uses=2]
	store <8 x i8> %tmp25, <8 x i8>* %B
	%tmp36 = load <8 x i8>* %A		; <<8 x i8>> [#uses=1]
	%tmp49 = tail call <8 x i8> @llvm.x86.mmx.paddus.b( <8 x i8> %tmp36, <8 x i8> %tmp25 )		; <<8 x i8>> [#uses=2]
	store <8 x i8> %tmp49, <8 x i8>* %B
	%tmp58 = load <8 x i8>* %A		; <<8 x i8>> [#uses=1]
	%tmp61 = sub <8 x i8> %tmp58, %tmp49		; <<8 x i8>> [#uses=2]
	store <8 x i8> %tmp61, <8 x i8>* %B
	%tmp64 = load <8 x i8>* %A		; <<8 x i8>> [#uses=1]
	%tmp80 = tail call <8 x i8> @llvm.x86.mmx.psubs.b( <8 x i8> %tmp61, <8 x i8> %tmp64 )		; <<8 x i8>> [#uses=2]
	store <8 x i8> %tmp80, <8 x i8>* %A
	%tmp89 = load <8 x i8>* %B		; <<8 x i8>> [#uses=1]
	%tmp105 = tail call <8 x i8> @llvm.x86.mmx.psubus.b( <8 x i8> %tmp80, <8 x i8> %tmp89 )		; <<8 x i8>> [#uses=1]
	store <8 x i8> %tmp105, <8 x i8>* %A
        %tmp13 = load <8 x i8>* %A              ; <<8 x i8>> [#uses=1]
        %tmp16 = mul <8 x i8> %tmp13, %tmp105            ; <<8 x i8>> [#uses=1]
        store <8 x i8> %tmp16, <8 x i8>* %B
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

define void @baz(<2 x i32>* %A, <2 x i32>* %B) {
entry:
	%tmp1 = load <2 x i32>* %A		; <<2 x i32>> [#uses=1]
	%tmp3 = load <2 x i32>* %B		; <<2 x i32>> [#uses=1]
	%tmp4 = add <2 x i32> %tmp1, %tmp3		; <<2 x i32>> [#uses=2]
	store <2 x i32> %tmp4, <2 x i32>* %A
	%tmp9 = load <2 x i32>* %B		; <<2 x i32>> [#uses=1]
	%tmp10 = sub <2 x i32> %tmp4, %tmp9		; <<2 x i32>> [#uses=1]
	store <2 x i32> %tmp10, <2 x i32>* %B
        %tmp13 = load <2 x i32>* %A             ; <<2 x i32>> [#uses=1]
        %tmp16 = mul <2 x i32> %tmp13, %tmp10           ; <<2 x i32>> [#uses=1]
        store <2 x i32> %tmp16, <2 x i32>* %B
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

define void @bar(<4 x i16>* %A, <4 x i16>* %B) {
entry:
	%tmp5 = load <4 x i16>* %A		; <<4 x i16>> [#uses=1]
	%tmp7 = load <4 x i16>* %B		; <<4 x i16>> [#uses=1]
	%tmp8 = add <4 x i16> %tmp5, %tmp7		; <<4 x i16>> [#uses=2]
	store <4 x i16> %tmp8, <4 x i16>* %A
	%tmp14 = load <4 x i16>* %B		; <<4 x i16>> [#uses=1]
	%tmp25 = tail call <4 x i16> @llvm.x86.mmx.padds.w( <4 x i16> %tmp14, <4 x i16> %tmp8 )		; <<4 x i16>> [#uses=2]
	store <4 x i16> %tmp25, <4 x i16>* %B
	%tmp36 = load <4 x i16>* %A		; <<4 x i16>> [#uses=1]
	%tmp49 = tail call <4 x i16> @llvm.x86.mmx.paddus.w( <4 x i16> %tmp36, <4 x i16> %tmp25 )		; <<4 x i16>> [#uses=2]
	store <4 x i16> %tmp49, <4 x i16>* %B
	%tmp58 = load <4 x i16>* %A		; <<4 x i16>> [#uses=1]
	%tmp61 = sub <4 x i16> %tmp58, %tmp49		; <<4 x i16>> [#uses=2]
	store <4 x i16> %tmp61, <4 x i16>* %B
	%tmp64 = load <4 x i16>* %A		; <<4 x i16>> [#uses=1]
	%tmp80 = tail call <4 x i16> @llvm.x86.mmx.psubs.w( <4 x i16> %tmp61, <4 x i16> %tmp64 )		; <<4 x i16>> [#uses=2]
	store <4 x i16> %tmp80, <4 x i16>* %A
	%tmp89 = load <4 x i16>* %B		; <<4 x i16>> [#uses=1]
	%tmp105 = tail call <4 x i16> @llvm.x86.mmx.psubus.w( <4 x i16> %tmp80, <4 x i16> %tmp89 )		; <<4 x i16>> [#uses=1]
	store <4 x i16> %tmp105, <4 x i16>* %A
        %tmp22 = load <4 x i16>* %A             ; <<4 x i16>> [#uses=1]
        %tmp24 = tail call <4 x i16> @llvm.x86.mmx.pmulh.w( <4 x i16> %tmp22, <4 x i16> %tmp105 )                ; <<4 x i16>> [#uses=2]
        store <4 x i16> %tmp24, <4 x i16>* %A
        %tmp28 = load <4 x i16>* %B             ; <<4 x i16>> [#uses=1]
        %tmp33 = tail call <2 x i32> @llvm.x86.mmx.pmadd.wd( <4 x i16> %tmp24, <4 x i16> %tmp28 )               ; <<2 x i32>> [#uses=1]
        %tmp34 = bitcast <2 x i32> %tmp33 to <4 x i16>          ; <<4 x i16>> [#uses=1]
        store <4 x i16> %tmp34, <4 x i16>* %A
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

declare <4 x i16> @llvm.x86.mmx.padds.w(<4 x i16>, <4 x i16>)

declare <4 x i16> @llvm.x86.mmx.paddus.w(<4 x i16>, <4 x i16>)

declare <4 x i16> @llvm.x86.mmx.psubs.w(<4 x i16>, <4 x i16>)

declare <4 x i16> @llvm.x86.mmx.psubus.w(<4 x i16>, <4 x i16>)

declare <8 x i8> @llvm.x86.mmx.padds.b(<8 x i8>, <8 x i8>)

declare <8 x i8> @llvm.x86.mmx.paddus.b(<8 x i8>, <8 x i8>)

declare <8 x i8> @llvm.x86.mmx.psubs.b(<8 x i8>, <8 x i8>)

declare <8 x i8> @llvm.x86.mmx.psubus.b(<8 x i8>, <8 x i8>)

declare <4 x i16> @llvm.x86.mmx.pmulh.w(<4 x i16>, <4 x i16>)

declare <2 x i32> @llvm.x86.mmx.pmadd.wd(<4 x i16>, <4 x i16>)

declare void @llvm.x86.mmx.emms()
