char rcsid_queue[] = "$Id$";

#include "b.h"
#include <stdio.h>

Queue globalQ;

Queue
newQ()
{
	Queue q;

	q = (Queue) zalloc(sizeof(struct queue));
	assert(q);
	q->head = 0;
	q->tail = 0;

	return q;
}

void
addQ(q, ts) Queue q; Item_Set ts;
{
	List qe;

	assert(q);
	assert(ts);

	qe = newList(ts, 0);
	if (q->head) {
		assert(q->tail);
		q->tail->next = qe;
		q->tail = qe;
	} else {
		q->head = q->tail = qe;
	}
}

Item_Set
popQ(q) Queue q;
{
	List qe;
	Item_Set ts;

	assert(q);

	if (q->head) {
		qe = q->head;
		q->head = q->head->next;
		ts = (Item_Set) qe->x;
		zfree(qe);
		return ts;
	} else {
		return 0;
	}
}

void
dumpQ(q) Queue q;
{
	printf("Begin Queue\n");
	foreachList((ListFn)dumpItem_Set, q->head);
	printf("End Queue\n");
}
