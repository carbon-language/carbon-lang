/*
 * Header file: wait.h
 *
 * Description:
 *	This header file includes the headers needed for the wait() system
 *	call.
 */

#ifndef _CONFIG_SYS_WAIT_H
#define _CONFIG_SYS_WAIT_H

#include "Config/config.h"

#ifdef HAVE_SYS_WAIT_H
#include <sys/wait.h>
#endif

#endif

