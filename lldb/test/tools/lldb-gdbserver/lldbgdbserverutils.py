"""Module for supporting unit testing of the lldb-gdbserver debug monitor exe.
"""

import os
import os.path
import re
import select
import time


def _get_debug_monitor_from_lldb(lldb_exe, debug_monitor_basename):
    """Return the debug monitor exe path given the lldb exe path.

    This method attempts to construct a valid debug monitor exe name
    from a given lldb exe name.  It will return None if the synthesized
    debug monitor name is not found to exist.

    The debug monitor exe path is synthesized by taking the directory
    of the lldb exe, and replacing the portion of the base name that
    matches "lldb" (case insensitive) and replacing with the value of
    debug_monitor_basename.

    Args:
        lldb_exe: the path to an lldb executable.

        debug_monitor_basename: the base name portion of the debug monitor
            that will replace 'lldb'.

    Returns:
        A path to the debug monitor exe if it is found to exist; otherwise,
        returns None.

    """

    exe_dir = os.path.dirname(lldb_exe)
    exe_base = os.path.basename(lldb_exe)

    # we'll rebuild the filename by replacing lldb with
    # the debug monitor basename, keeping any prefix or suffix in place.
    regex = re.compile(r"lldb", re.IGNORECASE)
    new_base = regex.sub(debug_monitor_basename, exe_base)

    debug_monitor_exe = os.path.join(exe_dir, new_base)
    if os.path.exists(debug_monitor_exe):
        return debug_monitor_exe
    else:
        return None


def get_lldb_gdbserver_exe():
    """Return the lldb-gdbserver exe path.

    Returns:
        A path to the lldb-gdbserver exe if it is found to exist; otherwise,
        returns None.
    """
    lldb_exe = os.environ["LLDB_EXEC"]
    if not lldb_exe:
        return None
    else:
        return _get_debug_monitor_from_lldb(lldb_exe, "lldb-gedbserver")

def get_debugserver_exe():
    """Return the debugserver exe path.

    Returns:
        A path to the debugserver exe if it is found to exist; otherwise,
        returns None.
    """
    lldb_exe = os.environ["LLDB_EXEC"]
    if not lldb_exe:
        return None
    else:
        return _get_debug_monitor_from_lldb(lldb_exe, "debugserver")


_LOG_LINE_REGEX = re.compile(r'^(lldb-gdbserver|debugserver)\s+<\s*(\d+)>' +
    '\s+(read|send)\s+packet:\s+(.+)$')


def _is_packet_lldb_gdbserver_input(packet_type, llgs_input_is_read):
    """Return whether a given packet is input for lldb-gdbserver.

    Args:
        packet_type: a string indicating 'send' or 'receive', from a
            gdbremote packet protocol log.

        llgs_input_is_read: true if lldb-gdbserver input (content sent to
            lldb-gdbserver) is listed as 'read' or 'send' in the packet
            log entry.

    Returns:
        True if the packet should be considered input for lldb-gdbserver; False
        otherwise.
    """
    if packet_type == 'read':
        # when llgs is the read side, then a read packet is meant for
        # input to llgs (when captured from the llgs/debugserver exe).
        return llgs_input_is_read
    elif packet_type == 'send':
        # when llgs is the send side, then a send packet is meant to
        # be input to llgs (when captured from the lldb exe).
        return not llgs_input_is_read
    else:
        # don't understand what type of packet this is
        raise "Unknown packet type: {}".format(packet_type)


_STRIP_CHECKSUM_REGEX = re.compile(r'#[0-9a-fA-F]{2}$')

def assert_packets_equal(asserter, actual_packet, expected_packet):
    # strip off the checksum digits of the packet.  When we're in
    # no-ack mode, the # checksum is ignored, and should not be cause
    # for a mismatched packet.
    actual_stripped = _STRIP_CHECKSUM_REGEX.sub('', actual_packet)
    expected_stripped = _STRIP_CHECKSUM_REGEX.sub('', expected_packet)
    asserter.assertEqual(actual_stripped, expected_stripped)


_GDB_REMOTE_PACKET_REGEX = re.compile(r'^\$([^\#]*)#[0-9a-fA-F]{2}')

def expect_lldb_gdbserver_replay(
    asserter,
    sock,
    log_lines,
    read_is_llgs_input,
    timeout_seconds,
    logger=None):
    """Replay socket communication with lldb-gdbserver and verify responses.

    Args:
        asserter: the object providing assertEqual(first, second, msg=None), e.g. TestCase instance.

        sock: the TCP socket connected to the lldb-gdbserver exe.

        log_lines: an array of text lines output from packet logging
           within lldb or lldb-gdbserver. Should look something like
           this:

           lldb-gdbserver <  19> read packet: $QStartNoAckMode#b0
           lldb-gdbserver <   1> send packet: +
           lldb-gdbserver <   6> send packet: $OK#9a
           lldb-gdbserver <   1> read packet: +

        read_is_llgs_input: True if packet logs list lldb-gdbserver
           input as the read side. False if lldb-gdbserver input is
           listed as the send side. Logs could be generated from
           either side, and this just allows supporting either one.

        timeout_seconds: any response taking more than this number of
           seconds will cause an exception to be raised.

    Returns:
        None if no issues.  Raises an exception if the expected communication does not
        occur.

    """
    received_lines = []
    receive_buffer = ''

    for packet in log_lines:
        if logger:
            logger.debug("processing log line: {}".format(packet))
        match = _LOG_LINE_REGEX.match(packet)
        if match:
            playback_packet = match.group(4)
            if _is_packet_lldb_gdbserver_input(
                    match.group(3),
                    read_is_llgs_input):
                # handle as something to send to lldb-gdbserver on
                # socket.
                if logger:
                    logger.info("sending packet to llgs: {}".format(playback_packet))
                sock.sendall(playback_packet)
            else:
                # expect it as output from lldb-gdbserver received
                # from socket.
                if logger:
                    logger.info("receiving packet from llgs, should match: {}".format(playback_packet))
                start_time = time.time()
                timeout_time = start_time + timeout_seconds

                # while we don't have a complete line of input, wait
                # for it from socket.
                while len(received_lines) < 1:
                    # check for timeout
                    if time.time() > timeout_time:
                        raise Exception(
                            'timed out after {} seconds while waiting for llgs to respond with: {}, currently received: {}'.format(
                                timeout_seconds, playback_packet, receive_buffer))
                    can_read, _, _ = select.select(
                        [sock], [], [], 0)
                    if can_read and sock in can_read:
                        new_bytes = sock.recv(4096)
                        if new_bytes and len(new_bytes) > 0:
                            # read the next bits from the socket
                            if logger:
                                logger.debug("llgs responded with bytes: {}".format(new_bytes))
                            receive_buffer += new_bytes

                            # parse fully-formed packets into individual packets
                            has_more = len(receive_buffer) > 0
                            while has_more:
                                if len(receive_buffer) <= 0:
                                    has_more = False
                                # handle '+' ack
                                elif receive_buffer[0] == '+':
                                    received_lines.append('+')
                                    receive_buffer = receive_buffer[1:]
                                    if logger:
                                        logger.debug('parsed packet from llgs: +, new receive_buffer: {}'.format(receive_buffer))
                                else:
                                    packet_match = _GDB_REMOTE_PACKET_REGEX.match(receive_buffer)
                                    if packet_match:
                                        received_lines.append(packet_match.group(0))
                                        receive_buffer = receive_buffer[len(packet_match.group(0)):]
                                        if logger:
                                            logger.debug('parsed packet from llgs: {}, new receive_buffer: {}'.format(packet_match.group(0), receive_buffer))
                                    else:
                                        has_more = False

                # got a line - now try to match it against expected line
                if len(received_lines) > 0:
                    received_packet = received_lines.pop(0)
                    assert_packets_equal(asserter, received_packet, playback_packet)

    return None


def gdbremote_hex_encode_string(str):
    output = ''
    for c in str:
        output += '{0:02x}'.format(ord(c))
    return output


def gdbremote_packet_encode_string(str):
    checksum = 0
    for c in str:
        checksum += ord(c)
    return '$' + str + '#{0:02x}'.format(checksum % 256)


if __name__ == '__main__':
    EXE_PATH = get_lldb_gdbserver_exe()
    if EXE_PATH:
        print "lldb-gdbserver path detected: {}".format(EXE_PATH)
    else:
        print "lldb-gdbserver could not be found"
