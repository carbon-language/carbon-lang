; RUN: llvm-as < %s | opt -loop-extract -disable-output

	%struct.node_t = type { double*, %struct.node_t*, %struct.node_t**, double**, double*, int, int }
	%struct.table_t = type { [1 x %struct.node_t**], [1 x %struct.node_t**] }

implementation   ; Functions:

void %make_tables() {
entry:
	%tmp.0.i = malloc %struct.node_t		; <%struct.node_t*> [#uses=1]
	br bool false, label %no_exit.i, label %loopexit.i

no_exit.i:		; preds = %entry, %no_exit.i
	%prev_node.0.i.1 = phi %struct.node_t* [ %tmp.16.i, %no_exit.i ], [ %tmp.0.i, %entry ]		; <%struct.node_t*> [#uses=0]
	%tmp.16.i = malloc %struct.node_t		; <%struct.node_t*> [#uses=2]
	br bool false, label %no_exit.i, label %loopexit.i

loopexit.i:		; preds = %entry, %no_exit.i
	%cur_node.0.i.0 = phi %struct.node_t* [ null, %entry ], [ %tmp.16.i, %no_exit.i ]		; <%struct.node_t*> [#uses=0]
	ret void
}
