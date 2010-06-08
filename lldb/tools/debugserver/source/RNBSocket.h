//===-- RNBSocket.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 12/12/07.
//
//===----------------------------------------------------------------------===//

#ifndef __RNBSocket_h__
#define __RNBSocket_h__

#include "RNBDefs.h"
#include <sys/socket.h>
#include <sys/types.h>
#include <string>
#include "DNBTimer.h"

class RNBSocket
{
public:

    RNBSocket () :
        m_conn_port (-1),
        m_conn_port_from_lockdown (false),
        m_timer (true)      // Make a thread safe timer
    {
    }
    ~RNBSocket (void)
    {
        Disconnect (false);
    }

    rnb_err_t Listen (in_port_t listen_port_num);
#if defined (__arm__)
    rnb_err_t ConnectToService();
#endif
    rnb_err_t OpenFile (const char *path);
    rnb_err_t Disconnect (bool save_errno);
    rnb_err_t Read (std::string &p);
    rnb_err_t Write (const void *buffer, size_t length);

    bool IsConnected () const { return m_conn_port != -1; }
    void SaveErrno (int curr_errno);
    DNBTimer& Timer() { return m_timer; }

    static int SetSocketOption(int fd, int level, int option_name, int option_value);
private:
    // Outlaw some constructors
    RNBSocket (const RNBSocket &);

protected:
    rnb_err_t ClosePort (int& fd, bool save_errno);

    int m_conn_port;    // Socket we use to communicate once conn established
    bool m_conn_port_from_lockdown;
    DNBTimer m_timer;
};


#endif // #ifndef __RNBSocket_h__
