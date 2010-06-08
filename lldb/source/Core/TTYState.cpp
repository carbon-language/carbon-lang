//===-- TTYState.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/TTYState.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/signal.h>

using namespace lldb_private;

//----------------------------------------------------------------------
// Default constructor
//----------------------------------------------------------------------
TTYState::TTYState() :
    m_fd(-1),
    m_tflags(-1),
    m_ttystate_err(-1),
    m_ttystate(),
    m_process_group(-1)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
TTYState::~TTYState()
{
}

//----------------------------------------------------------------------
// Save the current state of the TTY for the file descriptor "fd"
// and if "save_process_group" is true, attempt to save the process
// group info for the TTY.
//----------------------------------------------------------------------
bool
TTYState::Save (int fd, bool save_process_group)
{
    if (fd >= 0 && ::isatty (fd))
    {
        m_fd = fd;
        m_tflags = ::fcntl (fd, F_GETFL, 0);
        m_ttystate_err = ::tcgetattr (fd, &m_ttystate);
        if (save_process_group)
            m_process_group = ::tcgetpgrp (0);
        else
            m_process_group = -1;
    }
    else
    {
        m_fd = -1;
        m_tflags = -1;
        m_ttystate_err = -1;
        m_process_group = -1;
    }
    return m_ttystate_err == 0;
}

//----------------------------------------------------------------------
// Restore the state of the TTY using the cached values from a
// previous call to Save().
//----------------------------------------------------------------------
bool
TTYState::Restore () const
{
    int result = 0;
    if (IsValid())
    {
        if (TFlagsIsValid())
            result = fcntl (m_fd, F_SETFL, m_tflags);

        if (TTYStateIsValid())
            result = tcsetattr (m_fd, TCSANOW, &m_ttystate);

        if (ProcessGroupIsValid())
        {
            // Save the original signal handler.
            void (*saved_sigttou_callback) (int) = NULL;
            saved_sigttou_callback = (void (*)(int)) signal (SIGTTOU, SIG_IGN);
            // Set the process group
            result = tcsetpgrp (m_fd, m_process_group);
            // Restore the original signal handler.
            signal (SIGTTOU, saved_sigttou_callback);
        }
        return true;
    }
    return false;
}




//----------------------------------------------------------------------
// Returns true if this object has valid saved TTY state settings
// that can be used to restore a previous state.
//----------------------------------------------------------------------
bool
TTYState::IsValid() const
{
    return (m_fd >= 0) && (TFlagsIsValid() || TTYStateIsValid());
}

//----------------------------------------------------------------------
// Returns true if m_tflags is valid
//----------------------------------------------------------------------
bool
TTYState::TFlagsIsValid() const
{
    return m_tflags != -1;
}

//----------------------------------------------------------------------
// Returns true if m_ttystate is valid
//----------------------------------------------------------------------
bool
TTYState::TTYStateIsValid() const
{
    return m_ttystate_err == 0;
}

//----------------------------------------------------------------------
// Returns true if m_process_group is valid
//----------------------------------------------------------------------
bool
TTYState::ProcessGroupIsValid() const
{
    return m_process_group != -1;
}

//------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------
TTYStateSwitcher::TTYStateSwitcher () :
    m_currentState(UINT32_MAX)
{
}

//------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------
TTYStateSwitcher::~TTYStateSwitcher ()
{
}

//------------------------------------------------------------------
// Returns the number of states that this switcher contains
//------------------------------------------------------------------
uint32_t
TTYStateSwitcher::GetNumberOfStates() const
{
    return sizeof(m_ttystates)/sizeof(TTYState);
}

//------------------------------------------------------------------
// Restore the state at index "idx".
//
// Returns true if the restore was successful, false otherwise.
//------------------------------------------------------------------
bool
TTYStateSwitcher::Restore (uint32_t idx) const
{
    const uint32_t num_states = GetNumberOfStates();
    if (idx >= num_states)
        return false;

    // See if we already are in this state?
    if (m_currentState < num_states && (idx == m_currentState) && m_ttystates[idx].IsValid())
        return true;

    // Set the state to match the index passed in and only update the
    // current state if there are no errors.
    if (m_ttystates[idx].Restore())
    {
        m_currentState = idx;
        return true;
    }

    // We failed to set the state. The tty state was invalid or not
    // initialized.
    return false;
}

//------------------------------------------------------------------
// Save the state at index "idx" for file descriptor "fd" and
// save the process group if requested.
//
// Returns true if the restore was successful, false otherwise.
//------------------------------------------------------------------
bool
TTYStateSwitcher::Save (uint32_t idx, int fd, bool save_process_group)
{
    const uint32_t num_states = GetNumberOfStates();
    if (idx < num_states)
        return m_ttystates[idx].Save(fd, save_process_group);
    return false;
}


