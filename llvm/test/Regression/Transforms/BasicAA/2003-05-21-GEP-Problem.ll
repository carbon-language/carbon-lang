; RUN: as < %s | opt -licm -disable-output
target endian = big
target pointersize = 64
	%struct..apr_array_header_t = type { %struct..apr_pool_t*, int, int, int, sbyte* }
	%struct..apr_pool_t = type opaque
	%struct..apr_table_t = type { %struct..apr_array_header_t, uint, [32 x int], [32 x int] }

implementation   ; Functions:

void %table_reindex(%struct..apr_table_t* %t.1) {		; No predecessors!
	br label %loopentry

loopentry:		; preds = %0, %no_exit
	%tmp.101 = getelementptr %struct..apr_table_t* %t.1, long 0, ubyte 0, ubyte 2		; <int*> [#uses=1]
	%tmp.11 = load int* %tmp.101		; <int> [#uses=0]
	br bool false, label %no_exit, label %UnifiedExitNode

no_exit:		; preds = %loopentry
	%tmp.25 = cast int 0 to long		; <long> [#uses=1]
	%tmp.261 = getelementptr %struct..apr_table_t* %t.1, long 0, ubyte 3, long %tmp.25		; <int*> [#uses=1]
	store int 0, int* %tmp.261
	br label %loopentry

UnifiedExitNode:		; preds = %loopentry
	ret void
}
