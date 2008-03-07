; RUN: llvm-as < %s > %t.bc
; RUN: llvm-as < %p/2008-03-07-DroppedSection_b.ll > %t2.bc
; RUN: llvm-ld -r -disable-opt %t.bc %t2.bc -o %t3.bc
; RUN: llvm-dis < %t3.bc | grep ".data.init_task"

; ModuleID = 't.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-pc-linux-gnu"
	%struct.ViceFid = type { i32, i32, i32 }
	%struct.__kernel_fsid_t = type { [2 x i32] }
	%struct.address_space = type { %struct.list_head, %struct.list_head, %struct.list_head, i32, %struct.address_space_operations*, %struct.inode*, %struct.vm_area_struct*, %struct.vm_area_struct*, %struct.reiserfs_proc_info_data_t, i32 }
	%struct.address_space_operations = type { i32 (%struct.page*)*, i32 (%struct.file*, %struct.page*)*, i32 (%struct.page*)*, i32 (%struct.file*, %struct.page*, i32, i32)*, i32 (%struct.file*, %struct.page*, i32, i32)*, i32 (%struct.address_space*, i32)*, i32 (%struct.page*, i32)*, i32 (%struct.page*, i32)*, i32 (i32, %struct.inode*, %struct.kiobuf*, i32, i32)*, i32 (i32, %struct.file*, %struct.kiobuf*, i32, i32)*, void (%struct.page*)* }
	%struct.affs_bm_info = type { i32, i32 }
	%struct.atomic_t = type { i32 }
	%struct.block_device = type { %struct.list_head, %struct.atomic_t, %struct.inode*, i16, i32, %struct.block_device_operations*, %struct.semaphore, %struct.list_head }
	%struct.block_device_operations = type { i32 (%struct.inode*, %struct.file*)*, i32 (%struct.inode*, %struct.file*)*, i32 (%struct.inode*, %struct.file*, i32, i32)*, i32 (i16)*, i32 (i16)*, %struct.module* }
	%struct.buffer_head = type { %struct.buffer_head*, i32, i16, i16, i16, %struct.atomic_t, i16, i32, i32, %struct.buffer_head*, %struct.buffer_head*, %struct.buffer_head*, %struct.buffer_head*, %struct.buffer_head**, i8*, %struct.page*, void (%struct.buffer_head*, i32)*, i8*, i32, %struct.wait_queue_head_t, %struct.list_head }
	%struct.char_device = type { %struct.list_head, %struct.atomic_t, i16, %struct.atomic_t, %struct.semaphore }
	%struct.completion = type { i32, %struct.wait_queue_head_t }
	%struct.dentry = type { %struct.atomic_t, i32, %struct.inode*, %struct.dentry*, %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, i32, %struct.qstr, i32, %struct.dentry_operations*, %struct.super_block*, i32, i8*, [16 x i8] }
	%struct.dentry_operations = type { i32 (%struct.dentry*, i32)*, i32 (%struct.dentry*, %struct.qstr*)*, i32 (%struct.dentry*, %struct.qstr*, %struct.qstr*)*, i32 (%struct.dentry*)*, void (%struct.dentry*)*, void (%struct.dentry*, %struct.inode*)* }
	%struct.dnotify_struct = type { %struct.dnotify_struct*, i32, i32, %struct.file*, %struct.files_struct* }
	%struct.dquot = type { %struct.list_head, %struct.list_head, %struct.list_head, %struct.wait_queue_head_t, %struct.wait_queue_head_t, i32, i32, %struct.super_block*, i32, i16, i64, i16, i16, i32, %struct.mem_dqblk }
	%struct.dquot_operations = type { void (%struct.inode*, i32)*, void (%struct.inode*)*, i32 (%struct.inode*, i64, i32)*, i32 (%struct.inode*, i32)*, void (%struct.inode*, i64)*, void (%struct.inode*, i32)*, i32 (%struct.inode*, %struct.iattr*)*, i32 (%struct.dquot*)* }
	%struct.e820entry = type { i64, i64, i32 }
	%struct.exec_domain = type { i8*, void (i32, %struct.pt_regs*)*, i8, i8, i32*, i32*, %struct.map_segment*, %struct.map_segment*, %struct.map_segment*, %struct.map_segment*, %struct.module*, %struct.exec_domain* }
	%struct.fasync_struct = type { i32, i32, %struct.fasync_struct*, %struct.file* }
	%struct.fd_set = type { [32 x i32] }
	%struct.file = type { %struct.list_head, %struct.dentry*, %struct.vfsmount*, %struct.file_operations*, %struct.atomic_t, i32, i16, i64, i32, i32, i32, i32, i32, %struct.fown_struct, i32, i32, i32, i32, i8*, %struct.kiobuf*, i32 }
	%struct.file_lock = type { %struct.file_lock*, %struct.list_head, %struct.list_head, %struct.files_struct*, i32, %struct.wait_queue_head_t, %struct.file*, i8, i8, i64, i64, void (%struct.file_lock*)*, void (%struct.file_lock*)*, void (%struct.file_lock*)*, %struct.fasync_struct*, i32, { %struct.nfs_lock_info } }
	%struct.file_operations = type { %struct.module*, i64 (%struct.file*, i64, i32)*, i32 (%struct.file*, i8*, i32, i64*)*, i32 (%struct.file*, i8*, i32, i64*)*, i32 (%struct.file*, i8*, i32 (i8*, i8*, i32, i64, i32, i32)*)*, i32 (%struct.file*, %struct.poll_table_struct*)*, i32 (%struct.inode*, %struct.file*, i32, i32)*, i32 (%struct.file*, %struct.vm_area_struct*)*, i32 (%struct.inode*, %struct.file*)*, i32 (%struct.file*)*, i32 (%struct.inode*, %struct.file*)*, i32 (%struct.file*, %struct.dentry*, i32)*, i32 (i32, %struct.file*, i32)*, i32 (%struct.file*, i32, %struct.file_lock*)*, i32 (%struct.file*, %struct.iovec*, i32, i64*)*, i32 (%struct.file*, %struct.iovec*, i32, i64*)*, i32 (%struct.file*, %struct.page*, i32, i32, i64*, i32)*, i32 (%struct.file*, i32, i32, i32, i32)* }
	%struct.file_system_type = type { i8*, i32, %struct.super_block* (%struct.super_block*, i8*, i32)*, %struct.module*, %struct.file_system_type*, %struct.list_head }
	%struct.files_struct = type { %struct.atomic_t, %struct.reiserfs_proc_info_data_t, i32, i32, i32, %struct.file**, %struct.fd_set*, %struct.fd_set*, %struct.fd_set, %struct.fd_set, [32 x %struct.file*] }
	%struct.fown_struct = type { i32, i32, i32, i32 }
	%struct.fs_disk_quota = type { i8, i8, i16, i32, i64, i64, i64, i64, i64, i64, i32, i32, i16, i16, i32, i64, i64, i64, i32, i16, i16, [8 x i8] }
	%struct.fs_quota_stat = type { i8, i16, i8, %struct.e820entry, %struct.e820entry, i32, i32, i32, i32, i16, i16 }
	%struct.fs_struct = type { %struct.atomic_t, %struct.reiserfs_proc_info_data_t, i32, %struct.dentry*, %struct.dentry*, %struct.dentry*, %struct.vfsmount*, %struct.vfsmount*, %struct.vfsmount* }
	%struct.i387_fsave_struct = type { i32, i32, i32, i32, i32, i32, i32, [20 x i32], i32 }
	%struct.i387_fxsave_struct = type { i16, i16, i16, i16, i32, i32, i32, i32, i32, i32, [32 x i32], [32 x i32], [56 x i32] }
	%struct.i387_union = type { %struct.i387_fxsave_struct }
	%struct.iattr = type { i32, i16, i32, i32, i64, i32, i32, i32, i32 }
	%struct.if_dqblk = type { i64, i64, i64, i64, i64, i64, i64, i64, i32 }
	%struct.if_dqinfo = type { i64, i64, i32, i32 }
	%struct.inode = type { %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, i32, %struct.atomic_t, i16, i16, i16, i32, i32, i16, i64, i32, i32, i32, i32, i32, i32, i32, i16, %struct.semaphore, %struct.rw_semaphore, %struct.semaphore, %struct.inode_operations*, %struct.file_operations*, %struct.super_block*, %struct.wait_queue_head_t, %struct.file_lock*, %struct.address_space*, %struct.address_space, [2 x %struct.dquot*], %struct.list_head, %struct.pipe_inode_info*, %struct.block_device*, %struct.char_device*, i32, %struct.dnotify_struct*, i32, i32, i8, %struct.atomic_t, i32, i32, { %struct.nfs_inode_info } }
	%struct.inode_operations = type { i32 (%struct.inode*, %struct.dentry*, i32)*, %struct.dentry* (%struct.inode*, %struct.dentry*)*, i32 (%struct.dentry*, %struct.inode*, %struct.dentry*)*, i32 (%struct.inode*, %struct.dentry*)*, i32 (%struct.inode*, %struct.dentry*, i8*)*, i32 (%struct.inode*, %struct.dentry*, i32)*, i32 (%struct.inode*, %struct.dentry*)*, i32 (%struct.inode*, %struct.dentry*, i32, i32)*, i32 (%struct.inode*, %struct.dentry*, %struct.inode*, %struct.dentry*)*, i32 (%struct.dentry*, i8*, i32)*, i32 (%struct.dentry*, %struct.nameidata*)*, void (%struct.inode*)*, i32 (%struct.inode*, i32)*, i32 (%struct.dentry*)*, i32 (%struct.dentry*, %struct.iattr*)*, i32 (%struct.dentry*, %struct.iattr*)*, i32 (%struct.dentry*, i8*, i8*, i32, i32)*, i32 (%struct.dentry*, i8*, i8*, i32)*, i32 (%struct.dentry*, i8*, i32)*, i32 (%struct.dentry*, i8*)* }
	%struct.iovec = type { i8*, i32 }
	%struct.k_sigaction = type { %struct.sigaction }
	%struct.kern_ipc_perm = type { i32, i32, i32, i32, i32, i16, i32 }
	%struct.kiobuf = type opaque
	%struct.linux_binfmt = type { %struct.linux_binfmt*, %struct.module*, i32 (%struct.linux_binprm*, %struct.pt_regs*)*, i32 (%struct.file*)*, i32 (i32, %struct.pt_regs*, %struct.file*)*, i32, i32 (%struct.linux_binprm*, i8*)* }
	%struct.linux_binprm = type { [128 x i8], [32 x %struct.page*], i32, i32, %struct.file*, i32, i32, i32, i32, i32, i32, i32, i8*, i32, i32 }
	%struct.list_head = type { %struct.list_head*, %struct.list_head* }
	%struct.llva_fp_state_t = type { [7 x i32], [20 x i32] }
	%struct.llva_icontext_t = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32*, i32 }
	%struct.llva_sigcontext = type { %struct.llva_icontext_t, %struct.llva_fp_state_t, i32, i32, i32, i32, [1 x i32], i8* }
	%struct.map_segment = type opaque
	%struct.mem_dqblk = type { i32, i32, i64, i32, i32, i32, i32, i32 }
	%struct.mem_dqinfo = type { %struct.quota_format_type*, i32, i32, i32, { %struct.ViceFid } }
	%struct.mm_struct = type <{ %struct.vm_area_struct*, %struct.rb_root_t, %struct.vm_area_struct*, %struct.atomic_t*, %struct.atomic_t, %struct.atomic_t, i32, %struct.rw_semaphore, %struct.reiserfs_proc_info_data_t, %struct.list_head, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, [3 x i8], %struct.iovec }>
	%struct.module = type { i32, %struct.module*, i8*, i32, { %struct.atomic_t }, i32, i32, i32, %struct.module_symbol*, %struct.module_ref*, %struct.module_ref*, i32 ()*, void ()*, %struct.affs_bm_info*, %struct.affs_bm_info*, %struct.module_persist*, %struct.module_persist*, i32 ()*, i32, i8*, i8*, i8*, i8*, i8* }
	%struct.module_persist = type opaque
	%struct.module_ref = type { %struct.module*, %struct.module*, %struct.module_ref* }
	%struct.module_symbol = type { i32, i8* }
	%struct.nameidata = type { %struct.dentry*, %struct.vfsmount*, %struct.qstr, i32, i32 }
	%struct.namespace = type opaque
	%struct.ncp_mount_data_kernel = type { i32, i32, i32, i32, i32, i32, i32, [17 x i8], i32, i32, i16, i16 }
	%struct.ncp_server = type { %struct.ncp_mount_data_kernel, [258 x i8], %struct.file*, i8, i8, i16, i8, i8, i32, i32, i32, i8*, i32, %struct.semaphore, i32, i32, i32, i32, i32, i32, [8 x i8], [16 x i8], { i32, i32, i8*, i32 }, %struct.module_symbol, %struct.nls_table*, %struct.nls_table*, i32, i32 }
	%struct.nfs_fh = type { i16, [64 x i8] }
	%struct.nfs_inode_info = type { i64, %struct.nfs_fh, i16, i32, i64, i64, i64, i32, i32, i32, [2 x i32], %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, i32, i32, i32, i32, %struct.rpc_cred* }
	%struct.nfs_lock_info = type { i32, i32, %struct.nlm_host* }
	%struct.nlm_host = type opaque
	%struct.nls_table = type opaque
	%struct.page = type { %struct.list_head, %struct.address_space*, i32, %struct.page*, %struct.atomic_t, i32, %struct.list_head, %struct.page**, %struct.buffer_head* }
	%struct.pipe_inode_info = type { %struct.wait_queue_head_t, i8*, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.poll_table_struct = type opaque
	%struct.proc_dir_entry = type { i16, i16, i8*, i16, i16, i32, i32, i32, %struct.inode_operations*, %struct.file_operations*, i32 (i8*, i8**, i32, i32)*, %struct.module*, %struct.proc_dir_entry*, %struct.proc_dir_entry*, %struct.proc_dir_entry*, i8*, i32 (i8*, i8**, i32, i32, i32*, i8*)*, i32 (%struct.file*, i8*, i32, i8*)*, %struct.atomic_t, i32, i16 }
	%struct.pt_regs = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.qstr = type { i8*, i32, i32 }
	%struct.quota_format_ops = type { i32 (%struct.super_block*, i32)*, i32 (%struct.super_block*, i32)*, i32 (%struct.super_block*, i32)*, i32 (%struct.super_block*, i32)*, i32 (%struct.dquot*)*, i32 (%struct.dquot*)* }
	%struct.quota_format_type = type { i32, %struct.quota_format_ops*, %struct.module*, %struct.quota_format_type* }
	%struct.quota_info = type { i32, %struct.semaphore, %struct.semaphore, [2 x %struct.file*], [2 x %struct.mem_dqinfo], [2 x %struct.quota_format_ops*] }
	%struct.quotactl_ops = type { i32 (%struct.super_block*, i32, i32, i8*)*, i32 (%struct.super_block*, i32)*, i32 (%struct.super_block*, i32)*, i32 (%struct.super_block*, i32, %struct.if_dqinfo*)*, i32 (%struct.super_block*, i32, %struct.if_dqinfo*)*, i32 (%struct.super_block*, i32, i32, %struct.if_dqblk*)*, i32 (%struct.super_block*, i32, i32, %struct.if_dqblk*)*, i32 (%struct.super_block*, %struct.fs_quota_stat*)*, i32 (%struct.super_block*, i32, i32)*, i32 (%struct.super_block*, i32, i32, %struct.fs_disk_quota*)*, i32 (%struct.super_block*, i32, i32, %struct.fs_disk_quota*)* }
	%struct.rb_node_s = type { %struct.rb_node_s*, i32, %struct.rb_node_s*, %struct.rb_node_s* }
	%struct.rb_root_t = type { %struct.rb_node_s* }
	%struct.reiserfs_proc_info_data_t = type {  }
	%struct.revectored_struct = type { [8 x i32] }
	%struct.rpc_cred = type opaque
	%struct.rw_semaphore = type { i32, %struct.reiserfs_proc_info_data_t, %struct.list_head }
	%struct.sem_array = type { %struct.kern_ipc_perm, i32, i32, %struct.affs_bm_info*, %struct.sem_queue*, %struct.sem_queue**, %struct.sem_undo*, i32 }
	%struct.sem_queue = type { %struct.sem_queue*, %struct.sem_queue**, %struct.task_struct*, %struct.sem_undo*, i32, i32, %struct.sem_array*, i32, %struct.sembuf*, i32, i32 }
	%struct.sem_undo = type { %struct.sem_undo*, %struct.sem_undo*, i32, i16* }
	%struct.semaphore = type { %struct.atomic_t, i32, %struct.wait_queue_head_t }
	%struct.sembuf = type { i16, i16, i16 }
	%struct.seq_file = type { i8*, i32, i32, i32, i64, %struct.semaphore, %struct.seq_operations*, i8* }
	%struct.seq_operations = type { i8* (%struct.seq_file*, i64*)*, void (%struct.seq_file*, i8*)*, i8* (%struct.seq_file*, i8*, i64*)*, i32 (%struct.seq_file*, i8*)* }
	%struct.sigaction = type { void (i32)*, i32, void ()*, %struct.__kernel_fsid_t }
	%struct.siginfo_t = type { i32, i32, i32, { [29 x i32] } }
	%struct.signal_struct = type { %struct.atomic_t, [64 x %struct.k_sigaction], %struct.reiserfs_proc_info_data_t }
	%struct.sigpending = type { %struct.sigqueue*, %struct.sigqueue**, %struct.__kernel_fsid_t }
	%struct.sigqueue = type { %struct.sigqueue*, %struct.siginfo_t }
	%struct.statfs = type { i32, i32, i32, i32, i32, i32, i32, %struct.__kernel_fsid_t, i32, [6 x i32] }
	%struct.super_block = type { %struct.list_head, i16, i32, i8, i8, i64, %struct.file_system_type*, %struct.super_operations*, %struct.dquot_operations*, %struct.quotactl_ops*, i32, i32, %struct.dentry*, %struct.rw_semaphore, %struct.semaphore, i32, %struct.atomic_t, %struct.list_head, %struct.list_head, %struct.list_head, %struct.block_device*, %struct.list_head, %struct.quota_info, { %struct.ncp_server }, %struct.semaphore, %struct.semaphore }
	%struct.super_operations = type { %struct.inode* (%struct.super_block*)*, void (%struct.inode*)*, void (%struct.inode*)*, void (%struct.inode*, i8*)*, void (%struct.inode*)*, void (%struct.inode*, i32)*, void (%struct.inode*)*, void (%struct.inode*)*, void (%struct.super_block*)*, void (%struct.super_block*)*, i32 (%struct.super_block*)*, void (%struct.super_block*)*, void (%struct.super_block*)*, i32 (%struct.super_block*, %struct.statfs*)*, i32 (%struct.super_block*, i32*, i8*)*, void (%struct.inode*)*, void (%struct.super_block*)*, %struct.dentry* (%struct.super_block*, i32*, i32, i32, i32)*, i32 (%struct.dentry*, i32*, i32*, i32)*, i32 (%struct.seq_file*, %struct.vfsmount*)* }
	%struct.task_struct = type <{ i32, i32, i32, %struct.atomic_t, %struct.exec_domain*, i32, i32, i32, i32, i32, i32, %struct.mm_struct*, i32, i32, i32, %struct.list_head, i32, %struct.task_struct*, %struct.task_struct*, %struct.mm_struct*, %struct.list_head, i32, i32, %struct.linux_binfmt*, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.list_head, %struct.task_struct*, %struct.task_struct**, %struct.wait_queue_head_t, %struct.completion*, i32, i32, i32, i32, i32, i32, i32, %struct.timer_list, %struct.fown_struct, i32, [1 x i32], [1 x i32], i32, i32, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, [32 x i32], i32, i32, i32, i8, [3 x i8], %struct.user_struct*, [11 x %struct.affs_bm_info], i16, [16 x i8], [2 x i8], i32, i32, %struct.tty_struct*, i32, %struct.sem_undo*, %struct.sem_queue*, [8 x i8], %struct.thread_struct, %struct.fs_struct*, %struct.files_struct*, %struct.namespace*, %struct.reiserfs_proc_info_data_t, %struct.signal_struct*, %struct.__kernel_fsid_t, %struct.sigpending, i32, i32, i32 (i8*)*, i8*, %struct.__kernel_fsid_t*, i32, i32, %struct.reiserfs_proc_info_data_t, i8*, %struct.llva_sigcontext*, i32, %struct.task_struct*, i32, %struct.llva_icontext_t, %struct.llva_fp_state_t, i32*, i32, i8*, [4 x i8*], [2 x i32] }>
	%struct.task_union = type <{ %struct.task_struct, [1632 x i32] }>
	%struct.termios = type { i32, i32, i32, i32, i8, [19 x i8] }
	%struct.thread_struct = type <{ i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, %struct.i387_union, %struct.vm86_struct*, i32, i32, i32, i32, i32, [33 x i32], [1 x i32] }>
	%struct.timer_list = type { %struct.list_head, i32, i32, void (i32)* }
	%struct.tq_struct = type { %struct.list_head, i32, void (i8*)*, i8* }
	%struct.tty_driver = type { i32, i8*, i8*, i32, i16, i16, i16, i16, i16, %struct.termios, i32, i32*, %struct.proc_dir_entry*, %struct.tty_driver*, %struct.tty_struct**, %struct.termios**, %struct.termios**, i8*, i32 (%struct.tty_struct*, %struct.file*)*, void (%struct.tty_struct*, %struct.file*)*, i32 (%struct.tty_struct*, i32, i8*, i32)*, void (%struct.tty_struct*, i8)*, void (%struct.tty_struct*)*, i32 (%struct.tty_struct*)*, i32 (%struct.tty_struct*)*, i32 (%struct.tty_struct*, %struct.file*, i32, i32)*, void (%struct.tty_struct*, %struct.termios*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*, i32)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*, i32)*, void (%struct.tty_struct*, i8)*, i32 (i8*, i8**, i32, i32, i32*, i8*)*, i32 (%struct.file*, i8*, i32, i8*)*, %struct.tty_driver*, %struct.tty_driver* }
	%struct.tty_flip_buffer = type { %struct.tq_struct, %struct.semaphore, i8*, i8*, i32, i32, [1024 x i8], [1024 x i8], [4 x i8] }
	%struct.tty_ldisc = type { i32, i8*, i32, i32, i32 (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, i32 (%struct.tty_struct*)*, i32 (%struct.tty_struct*, %struct.file*, i8*, i32)*, i32 (%struct.tty_struct*, %struct.file*, i8*, i32)*, i32 (%struct.tty_struct*, %struct.file*, i32, i32)*, void (%struct.tty_struct*, %struct.termios*)*, i32 (%struct.tty_struct*, %struct.file*, %struct.poll_table_struct*)*, void (%struct.tty_struct*, i8*, i8*, i32)*, i32 (%struct.tty_struct*)*, void (%struct.tty_struct*)* }
	%struct.tty_struct = type <{ i32, %struct.tty_driver, %struct.tty_ldisc, %struct.termios*, %struct.termios*, i32, i32, i16, [2 x i8], i32, i32, %struct.winsize, i8, i8, [2 x i8], %struct.tty_struct*, %struct.fasync_struct*, %struct.tty_flip_buffer, i32, i32, %struct.wait_queue_head_t, %struct.wait_queue_head_t, %struct.tq_struct, i8*, i8*, %struct.list_head, i32, i8, i8, i16, i32, i32, [8 x i32], i8*, i32, i32, i32, [128 x i32], i32, i32, i32, %struct.semaphore, %struct.semaphore, %struct.reiserfs_proc_info_data_t, %struct.tq_struct }>
	%struct.user_struct = type { %struct.atomic_t, %struct.atomic_t, %struct.atomic_t, %struct.user_struct*, %struct.user_struct**, i32 }
	%struct.vfsmount = type { %struct.list_head, %struct.vfsmount*, %struct.dentry*, %struct.dentry*, %struct.super_block*, %struct.list_head, %struct.list_head, %struct.atomic_t, i32, i8*, %struct.list_head }
	%struct.vm86_regs = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i16, i16, i32, i32, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16 }
	%struct.vm86_struct = type { %struct.vm86_regs, i32, i32, i32, %struct.revectored_struct, %struct.revectored_struct }
	%struct.vm_area_struct = type { %struct.mm_struct*, i32, i32, %struct.vm_area_struct*, %struct.atomic_t, i32, %struct.rb_node_s, %struct.vm_area_struct*, %struct.vm_area_struct**, %struct.vm_operations_struct*, i32, %struct.file*, i32, i8* }
	%struct.vm_operations_struct = type { void (%struct.vm_area_struct*)*, void (%struct.vm_area_struct*)*, %struct.page* (%struct.vm_area_struct*, i32, i32)* }
	%struct.wait_queue_head_t = type { %struct.reiserfs_proc_info_data_t, %struct.list_head }
	%struct.winsize = type { i16, i16, i16, i16 }
@init_mm = internal global %struct.mm_struct <{
    %struct.vm_area_struct* null, 
    %struct.rb_root_t zeroinitializer, 
    %struct.vm_area_struct* null, 
    %struct.atomic_t* getelementptr ([1024 x %struct.atomic_t]* @swapper_pg_dir, i32 0, i32 0), 
    %struct.atomic_t { i32 2 }, 
    %struct.atomic_t { i32 1 }, 
    i32 0, 
    %struct.rw_semaphore {
        i32 0, 
        %struct.reiserfs_proc_info_data_t zeroinitializer, 
        %struct.list_head { %struct.list_head* getelementptr (%struct.mm_struct* @init_mm, i32 0, i32 7, i32 2), %struct.list_head* getelementptr (%struct.mm_struct* @init_mm, i32 0, i32 7, i32 2) } }, 
    %struct.reiserfs_proc_info_data_t zeroinitializer, 
    %struct.list_head { %struct.list_head* getelementptr (%struct.mm_struct* @init_mm, i32 0, i32 9), %struct.list_head* getelementptr (%struct.mm_struct* @init_mm, i32 0, i32 9) }, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i8 0, 
    [3 x i8] zeroinitializer, 
    %struct.iovec zeroinitializer }>, align 32		; <%struct.mm_struct*> [#uses=3]
@init_task_union = global <{ <{ i32, i32, i32, %struct.atomic_t, %struct.exec_domain*, i32, i32, i32, i32, i32, i32, %struct.mm_struct*, i32, i32, i32, %struct.list_head, i32, %struct.task_struct*, %struct.task_struct*, %struct.mm_struct*, %struct.list_head, i32, i32, %struct.linux_binfmt*, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.list_head, %struct.task_struct*, %struct.task_struct**, %struct.wait_queue_head_t, %struct.completion*, i32, i32, i32, i32, i32, i32, i32, %struct.timer_list, %struct.fown_struct, i32, [1 x i32], [1 x i32], i32, i32, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, [32 x i32], i32, i32, i32, i8, [3 x i8], %struct.user_struct*, [11 x %struct.affs_bm_info], i16, [16 x i8], [2 x i8], i32, i32, %struct.tty_struct*, i32, %struct.sem_undo*, %struct.sem_queue*, [8 x i8], <{ i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, { %struct.i387_fsave_struct, [400 x i8] }, %struct.vm86_struct*, i32, i32, i32, i32, i32, [33 x i32], [1 x i32] }>, %struct.fs_struct*, %struct.files_struct*, %struct.namespace*, %struct.reiserfs_proc_info_data_t, %struct.signal_struct*, %struct.__kernel_fsid_t, %struct.sigpending, i32, i32, i32 (i8*)*, i8*, %struct.__kernel_fsid_t*, i32, i32, %struct.reiserfs_proc_info_data_t, i8*, %struct.llva_sigcontext*, i32, %struct.task_struct*, i32, %struct.llva_icontext_t, %struct.llva_fp_state_t, i32*, i32, i8*, [4 x i8*], [2 x i32] }>, [1632 x i32] }> <{ <{ i32, i32, i32, %struct.atomic_t, %struct.exec_domain*, i32, i32, i32, i32, i32, i32, %struct.mm_struct*, i32, i32, i32, %struct.list_head, i32, %struct.task_struct*, %struct.task_struct*, %struct.mm_struct*, %struct.list_head, i32, i32, %struct.linux_binfmt*, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.list_head, %struct.task_struct*, %struct.task_struct**, %struct.wait_queue_head_t, %struct.completion*, i32, i32, i32, i32, i32, i32, i32, %struct.timer_list, %struct.fown_struct, i32, [1 x i32], [1 x i32], i32, i32, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, [32 x i32], i32, i32, i32, i8, [3 x i8], %struct.user_struct*, [11 x %struct.affs_bm_info], i16, [16 x i8], [2 x i8], i32, i32, %struct.tty_struct*, i32, %struct.sem_undo*, %struct.sem_queue*, [8 x i8], <{ i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, { %struct.i387_fsave_struct, [400 x i8] }, %struct.vm86_struct*, i32, i32, i32, i32, i32, [33 x i32], [1 x i32] }>, %struct.fs_struct*, %struct.files_struct*, %struct.namespace*, %struct.reiserfs_proc_info_data_t, %struct.signal_struct*, %struct.__kernel_fsid_t, %struct.sigpending, i32, i32, i32 (i8*)*, i8*, %struct.__kernel_fsid_t*, i32, i32, %struct.reiserfs_proc_info_data_t, i8*, %struct.llva_sigcontext*, i32, %struct.task_struct*, i32, %struct.llva_icontext_t, %struct.llva_fp_state_t, i32*, i32, i8*, [4 x i8*], [2 x i32] }> <{
    i32 0, 
    i32 0, 
    i32 0, 
    %struct.atomic_t { i32 -1 }, 
    %struct.exec_domain* @default_exec_domain, 
    i32 0, 
    i32 0, 
    i32 -1, 
    i32 10, 
    i32 0, 
    i32 0, 
    %struct.mm_struct* null, 
    i32 0, 
    i32 -1, 
    i32 -1, 
    %struct.list_head { %struct.list_head* getelementptr (%struct.task_union* bitcast (<{ <{ i32, i32, i32, %struct.atomic_t, %struct.exec_domain*, i32, i32, i32, i32, i32, i32, %struct.mm_struct*, i32, i32, i32, %struct.list_head, i32, %struct.task_struct*, %struct.task_struct*, %struct.mm_struct*, %struct.list_head, i32, i32, %struct.linux_binfmt*, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.list_head, %struct.task_struct*, %struct.task_struct**, %struct.wait_queue_head_t, %struct.completion*, i32, i32, i32, i32, i32, i32, i32, %struct.timer_list, %struct.fown_struct, i32, [1 x i32], [1 x i32], i32, i32, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, [32 x i32], i32, i32, i32, i8, [3 x i8], %struct.user_struct*, [11 x %struct.affs_bm_info], i16, [16 x i8], [2 x i8], i32, i32, %struct.tty_struct*, i32, %struct.sem_undo*, %struct.sem_queue*, [8 x i8], <{ i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, { %struct.i387_fsave_struct, [400 x i8] }, %struct.vm86_struct*, i32, i32, i32, i32, i32, [33 x i32], [1 x i32] }>, %struct.fs_struct*, %struct.files_struct*, %struct.namespace*, %struct.reiserfs_proc_info_data_t, %struct.signal_struct*, %struct.__kernel_fsid_t, %struct.sigpending, i32, i32, i32 (i8*)*, i8*, %struct.__kernel_fsid_t*, i32, i32, %struct.reiserfs_proc_info_data_t, i8*, %struct.llva_sigcontext*, i32, %struct.task_struct*, i32, %struct.llva_icontext_t, %struct.llva_fp_state_t, i32*, i32, i8*, [4 x i8*], [2 x i32] }>, [1632 x i32] }>* @init_task_union to %struct.task_union*), i32 0, i32 0, i32 15), %struct.list_head* getelementptr (%struct.task_union* bitcast (<{ <{ i32, i32, i32, %struct.atomic_t, %struct.exec_domain*, i32, i32, i32, i32, i32, i32, %struct.mm_struct*, i32, i32, i32, %struct.list_head, i32, %struct.task_struct*, %struct.task_struct*, %struct.mm_struct*, %struct.list_head, i32, i32, %struct.linux_binfmt*, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.list_head, %struct.task_struct*, %struct.task_struct**, %struct.wait_queue_head_t, %struct.completion*, i32, i32, i32, i32, i32, i32, i32, %struct.timer_list, %struct.fown_struct, i32, [1 x i32], [1 x i32], i32, i32, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, [32 x i32], i32, i32, i32, i8, [3 x i8], %struct.user_struct*, [11 x %struct.affs_bm_info], i16, [16 x i8], [2 x i8], i32, i32, %struct.tty_struct*, i32, %struct.sem_undo*, %struct.sem_queue*, [8 x i8], <{ i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, { %struct.i387_fsave_struct, [400 x i8] }, %struct.vm86_struct*, i32, i32, i32, i32, i32, [33 x i32], [1 x i32] }>, %struct.fs_struct*, %struct.files_struct*, %struct.namespace*, %struct.reiserfs_proc_info_data_t, %struct.signal_struct*, %struct.__kernel_fsid_t, %struct.sigpending, i32, i32, i32 (i8*)*, i8*, %struct.__kernel_fsid_t*, i32, i32, %struct.reiserfs_proc_info_data_t, i8*, %struct.llva_sigcontext*, i32, %struct.task_struct*, i32, %struct.llva_icontext_t, %struct.llva_fp_state_t, i32*, i32, i8*, [4 x i8*], [2 x i32] }>, [1632 x i32] }>* @init_task_union to %struct.task_union*), i32 0, i32 0, i32 15) }, 
    i32 0, 
    %struct.task_struct* getelementptr (%struct.task_union* bitcast (<{ <{ i32, i32, i32, %struct.atomic_t, %struct.exec_domain*, i32, i32, i32, i32, i32, i32, %struct.mm_struct*, i32, i32, i32, %struct.list_head, i32, %struct.task_struct*, %struct.task_struct*, %struct.mm_struct*, %struct.list_head, i32, i32, %struct.linux_binfmt*, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.list_head, %struct.task_struct*, %struct.task_struct**, %struct.wait_queue_head_t, %struct.completion*, i32, i32, i32, i32, i32, i32, i32, %struct.timer_list, %struct.fown_struct, i32, [1 x i32], [1 x i32], i32, i32, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, [32 x i32], i32, i32, i32, i8, [3 x i8], %struct.user_struct*, [11 x %struct.affs_bm_info], i16, [16 x i8], [2 x i8], i32, i32, %struct.tty_struct*, i32, %struct.sem_undo*, %struct.sem_queue*, [8 x i8], <{ i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, { %struct.i387_fsave_struct, [400 x i8] }, %struct.vm86_struct*, i32, i32, i32, i32, i32, [33 x i32], [1 x i32] }>, %struct.fs_struct*, %struct.files_struct*, %struct.namespace*, %struct.reiserfs_proc_info_data_t, %struct.signal_struct*, %struct.__kernel_fsid_t, %struct.sigpending, i32, i32, i32 (i8*)*, i8*, %struct.__kernel_fsid_t*, i32, i32, %struct.reiserfs_proc_info_data_t, i8*, %struct.llva_sigcontext*, i32, %struct.task_struct*, i32, %struct.llva_icontext_t, %struct.llva_fp_state_t, i32*, i32, i8*, [4 x i8*], [2 x i32] }>, [1632 x i32] }>* @init_task_union to %struct.task_union*), i32 0, i32 0), 
    %struct.task_struct* getelementptr (%struct.task_union* bitcast (<{ <{ i32, i32, i32, %struct.atomic_t, %struct.exec_domain*, i32, i32, i32, i32, i32, i32, %struct.mm_struct*, i32, i32, i32, %struct.list_head, i32, %struct.task_struct*, %struct.task_struct*, %struct.mm_struct*, %struct.list_head, i32, i32, %struct.linux_binfmt*, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.list_head, %struct.task_struct*, %struct.task_struct**, %struct.wait_queue_head_t, %struct.completion*, i32, i32, i32, i32, i32, i32, i32, %struct.timer_list, %struct.fown_struct, i32, [1 x i32], [1 x i32], i32, i32, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, [32 x i32], i32, i32, i32, i8, [3 x i8], %struct.user_struct*, [11 x %struct.affs_bm_info], i16, [16 x i8], [2 x i8], i32, i32, %struct.tty_struct*, i32, %struct.sem_undo*, %struct.sem_queue*, [8 x i8], <{ i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, { %struct.i387_fsave_struct, [400 x i8] }, %struct.vm86_struct*, i32, i32, i32, i32, i32, [33 x i32], [1 x i32] }>, %struct.fs_struct*, %struct.files_struct*, %struct.namespace*, %struct.reiserfs_proc_info_data_t, %struct.signal_struct*, %struct.__kernel_fsid_t, %struct.sigpending, i32, i32, i32 (i8*)*, i8*, %struct.__kernel_fsid_t*, i32, i32, %struct.reiserfs_proc_info_data_t, i8*, %struct.llva_sigcontext*, i32, %struct.task_struct*, i32, %struct.llva_icontext_t, %struct.llva_fp_state_t, i32*, i32, i8*, [4 x i8*], [2 x i32] }>, [1632 x i32] }>* @init_task_union to %struct.task_union*), i32 0, i32 0), 
    %struct.mm_struct* @init_mm, 
    %struct.list_head zeroinitializer, 
    i32 0, 
    i32 0, 
    %struct.linux_binfmt* null, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i8 0, 
    [3 x i8] zeroinitializer, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    %struct.task_struct* getelementptr (%struct.task_union* bitcast (<{ <{ i32, i32, i32, %struct.atomic_t, %struct.exec_domain*, i32, i32, i32, i32, i32, i32, %struct.mm_struct*, i32, i32, i32, %struct.list_head, i32, %struct.task_struct*, %struct.task_struct*, %struct.mm_struct*, %struct.list_head, i32, i32, %struct.linux_binfmt*, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.list_head, %struct.task_struct*, %struct.task_struct**, %struct.wait_queue_head_t, %struct.completion*, i32, i32, i32, i32, i32, i32, i32, %struct.timer_list, %struct.fown_struct, i32, [1 x i32], [1 x i32], i32, i32, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, [32 x i32], i32, i32, i32, i8, [3 x i8], %struct.user_struct*, [11 x %struct.affs_bm_info], i16, [16 x i8], [2 x i8], i32, i32, %struct.tty_struct*, i32, %struct.sem_undo*, %struct.sem_queue*, [8 x i8], <{ i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, { %struct.i387_fsave_struct, [400 x i8] }, %struct.vm86_struct*, i32, i32, i32, i32, i32, [33 x i32], [1 x i32] }>, %struct.fs_struct*, %struct.files_struct*, %struct.namespace*, %struct.reiserfs_proc_info_data_t, %struct.signal_struct*, %struct.__kernel_fsid_t, %struct.sigpending, i32, i32, i32 (i8*)*, i8*, %struct.__kernel_fsid_t*, i32, i32, %struct.reiserfs_proc_info_data_t, i8*, %struct.llva_sigcontext*, i32, %struct.task_struct*, i32, %struct.llva_icontext_t, %struct.llva_fp_state_t, i32*, i32, i8*, [4 x i8*], [2 x i32] }>, [1632 x i32] }>* @init_task_union to %struct.task_union*), i32 0, i32 0), 
    %struct.task_struct* getelementptr (%struct.task_union* bitcast (<{ <{ i32, i32, i32, %struct.atomic_t, %struct.exec_domain*, i32, i32, i32, i32, i32, i32, %struct.mm_struct*, i32, i32, i32, %struct.list_head, i32, %struct.task_struct*, %struct.task_struct*, %struct.mm_struct*, %struct.list_head, i32, i32, %struct.linux_binfmt*, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.list_head, %struct.task_struct*, %struct.task_struct**, %struct.wait_queue_head_t, %struct.completion*, i32, i32, i32, i32, i32, i32, i32, %struct.timer_list, %struct.fown_struct, i32, [1 x i32], [1 x i32], i32, i32, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, [32 x i32], i32, i32, i32, i8, [3 x i8], %struct.user_struct*, [11 x %struct.affs_bm_info], i16, [16 x i8], [2 x i8], i32, i32, %struct.tty_struct*, i32, %struct.sem_undo*, %struct.sem_queue*, [8 x i8], <{ i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, { %struct.i387_fsave_struct, [400 x i8] }, %struct.vm86_struct*, i32, i32, i32, i32, i32, [33 x i32], [1 x i32] }>, %struct.fs_struct*, %struct.files_struct*, %struct.namespace*, %struct.reiserfs_proc_info_data_t, %struct.signal_struct*, %struct.__kernel_fsid_t, %struct.sigpending, i32, i32, i32 (i8*)*, i8*, %struct.__kernel_fsid_t*, i32, i32, %struct.reiserfs_proc_info_data_t, i8*, %struct.llva_sigcontext*, i32, %struct.task_struct*, i32, %struct.llva_icontext_t, %struct.llva_fp_state_t, i32*, i32, i8*, [4 x i8*], [2 x i32] }>, [1632 x i32] }>* @init_task_union to %struct.task_union*), i32 0, i32 0), 
    %struct.task_struct* null, 
    %struct.task_struct* null, 
    %struct.task_struct* null, 
    %struct.list_head { %struct.list_head* getelementptr (%struct.task_union* bitcast (<{ <{ i32, i32, i32, %struct.atomic_t, %struct.exec_domain*, i32, i32, i32, i32, i32, i32, %struct.mm_struct*, i32, i32, i32, %struct.list_head, i32, %struct.task_struct*, %struct.task_struct*, %struct.mm_struct*, %struct.list_head, i32, i32, %struct.linux_binfmt*, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.list_head, %struct.task_struct*, %struct.task_struct**, %struct.wait_queue_head_t, %struct.completion*, i32, i32, i32, i32, i32, i32, i32, %struct.timer_list, %struct.fown_struct, i32, [1 x i32], [1 x i32], i32, i32, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, [32 x i32], i32, i32, i32, i8, [3 x i8], %struct.user_struct*, [11 x %struct.affs_bm_info], i16, [16 x i8], [2 x i8], i32, i32, %struct.tty_struct*, i32, %struct.sem_undo*, %struct.sem_queue*, [8 x i8], <{ i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, { %struct.i387_fsave_struct, [400 x i8] }, %struct.vm86_struct*, i32, i32, i32, i32, i32, [33 x i32], [1 x i32] }>, %struct.fs_struct*, %struct.files_struct*, %struct.namespace*, %struct.reiserfs_proc_info_data_t, %struct.signal_struct*, %struct.__kernel_fsid_t, %struct.sigpending, i32, i32, i32 (i8*)*, i8*, %struct.__kernel_fsid_t*, i32, i32, %struct.reiserfs_proc_info_data_t, i8*, %struct.llva_sigcontext*, i32, %struct.task_struct*, i32, %struct.llva_icontext_t, %struct.llva_fp_state_t, i32*, i32, i8*, [4 x i8*], [2 x i32] }>, [1632 x i32] }>* @init_task_union to %struct.task_union*), i32 0, i32 0, i32 41), %struct.list_head* getelementptr (%struct.task_union* bitcast (<{ <{ i32, i32, i32, %struct.atomic_t, %struct.exec_domain*, i32, i32, i32, i32, i32, i32, %struct.mm_struct*, i32, i32, i32, %struct.list_head, i32, %struct.task_struct*, %struct.task_struct*, %struct.mm_struct*, %struct.list_head, i32, i32, %struct.linux_binfmt*, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.list_head, %struct.task_struct*, %struct.task_struct**, %struct.wait_queue_head_t, %struct.completion*, i32, i32, i32, i32, i32, i32, i32, %struct.timer_list, %struct.fown_struct, i32, [1 x i32], [1 x i32], i32, i32, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, [32 x i32], i32, i32, i32, i8, [3 x i8], %struct.user_struct*, [11 x %struct.affs_bm_info], i16, [16 x i8], [2 x i8], i32, i32, %struct.tty_struct*, i32, %struct.sem_undo*, %struct.sem_queue*, [8 x i8], <{ i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, { %struct.i387_fsave_struct, [400 x i8] }, %struct.vm86_struct*, i32, i32, i32, i32, i32, [33 x i32], [1 x i32] }>, %struct.fs_struct*, %struct.files_struct*, %struct.namespace*, %struct.reiserfs_proc_info_data_t, %struct.signal_struct*, %struct.__kernel_fsid_t, %struct.sigpending, i32, i32, i32 (i8*)*, i8*, %struct.__kernel_fsid_t*, i32, i32, %struct.reiserfs_proc_info_data_t, i8*, %struct.llva_sigcontext*, i32, %struct.task_struct*, i32, %struct.llva_icontext_t, %struct.llva_fp_state_t, i32*, i32, i8*, [4 x i8*], [2 x i32] }>, [1632 x i32] }>* @init_task_union to %struct.task_union*), i32 0, i32 0, i32 41) }, 
    %struct.task_struct* null, 
    %struct.task_struct** null, 
    %struct.wait_queue_head_t { %struct.reiserfs_proc_info_data_t zeroinitializer, %struct.list_head { %struct.list_head* getelementptr (%struct.task_union* bitcast (<{ <{ i32, i32, i32, %struct.atomic_t, %struct.exec_domain*, i32, i32, i32, i32, i32, i32, %struct.mm_struct*, i32, i32, i32, %struct.list_head, i32, %struct.task_struct*, %struct.task_struct*, %struct.mm_struct*, %struct.list_head, i32, i32, %struct.linux_binfmt*, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.list_head, %struct.task_struct*, %struct.task_struct**, %struct.wait_queue_head_t, %struct.completion*, i32, i32, i32, i32, i32, i32, i32, %struct.timer_list, %struct.fown_struct, i32, [1 x i32], [1 x i32], i32, i32, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, [32 x i32], i32, i32, i32, i8, [3 x i8], %struct.user_struct*, [11 x %struct.affs_bm_info], i16, [16 x i8], [2 x i8], i32, i32, %struct.tty_struct*, i32, %struct.sem_undo*, %struct.sem_queue*, [8 x i8], <{ i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, { %struct.i387_fsave_struct, [400 x i8] }, %struct.vm86_struct*, i32, i32, i32, i32, i32, [33 x i32], [1 x i32] }>, %struct.fs_struct*, %struct.files_struct*, %struct.namespace*, %struct.reiserfs_proc_info_data_t, %struct.signal_struct*, %struct.__kernel_fsid_t, %struct.sigpending, i32, i32, i32 (i8*)*, i8*, %struct.__kernel_fsid_t*, i32, i32, %struct.reiserfs_proc_info_data_t, i8*, %struct.llva_sigcontext*, i32, %struct.task_struct*, i32, %struct.llva_icontext_t, %struct.llva_fp_state_t, i32*, i32, i8*, [4 x i8*], [2 x i32] }>, [1632 x i32] }>* @init_task_union to %struct.task_union*), i32 0, i32 0, i32 44, i32 1), %struct.list_head* getelementptr (%struct.task_union* bitcast (<{ <{ i32, i32, i32, %struct.atomic_t, %struct.exec_domain*, i32, i32, i32, i32, i32, i32, %struct.mm_struct*, i32, i32, i32, %struct.list_head, i32, %struct.task_struct*, %struct.task_struct*, %struct.mm_struct*, %struct.list_head, i32, i32, %struct.linux_binfmt*, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.list_head, %struct.task_struct*, %struct.task_struct**, %struct.wait_queue_head_t, %struct.completion*, i32, i32, i32, i32, i32, i32, i32, %struct.timer_list, %struct.fown_struct, i32, [1 x i32], [1 x i32], i32, i32, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, [32 x i32], i32, i32, i32, i8, [3 x i8], %struct.user_struct*, [11 x %struct.affs_bm_info], i16, [16 x i8], [2 x i8], i32, i32, %struct.tty_struct*, i32, %struct.sem_undo*, %struct.sem_queue*, [8 x i8], <{ i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, { %struct.i387_fsave_struct, [400 x i8] }, %struct.vm86_struct*, i32, i32, i32, i32, i32, [33 x i32], [1 x i32] }>, %struct.fs_struct*, %struct.files_struct*, %struct.namespace*, %struct.reiserfs_proc_info_data_t, %struct.signal_struct*, %struct.__kernel_fsid_t, %struct.sigpending, i32, i32, i32 (i8*)*, i8*, %struct.__kernel_fsid_t*, i32, i32, %struct.reiserfs_proc_info_data_t, i8*, %struct.llva_sigcontext*, i32, %struct.task_struct*, i32, %struct.llva_icontext_t, %struct.llva_fp_state_t, i32*, i32, i8*, [4 x i8*], [2 x i32] }>, [1632 x i32] }>* @init_task_union to %struct.task_union*), i32 0, i32 0, i32 44, i32 1) } }, 
    %struct.completion* null, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    %struct.timer_list {
        %struct.list_head zeroinitializer, 
        i32 0, 
        i32 0, 
        void (i32)* @it_real_fn }, 
    %struct.fown_struct zeroinitializer, 
    i32 0, 
    [1 x i32] zeroinitializer, 
    [1 x i32] zeroinitializer, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i8 0, 
    [3 x i8] zeroinitializer, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    i32 0, 
    [32 x i32] zeroinitializer, 
    i32 -257, 
    i32 0, 
    i32 -1, 
    i8 0, 
    [3 x i8] zeroinitializer, 
    %struct.user_struct* @root_user, 
    [11 x %struct.affs_bm_info] [ %struct.affs_bm_info { i32 -1, i32 -1 }, %struct.affs_bm_info { i32 -1, i32 -1 }, %struct.affs_bm_info { i32 -1, i32 -1 }, %struct.affs_bm_info { i32 8388608, i32 -1 }, %struct.affs_bm_info { i32 0, i32 -1 }, %struct.affs_bm_info { i32 -1, i32 -1 }, %struct.affs_bm_info zeroinitializer, %struct.affs_bm_info { i32 1024, i32 1024 }, %struct.affs_bm_info { i32 -1, i32 -1 }, %struct.affs_bm_info { i32 -1, i32 -1 }, %struct.affs_bm_info { i32 -1, i32 -1 } ], 
    i16 0, 
    [16 x i8] c"swapper\00\00\00\00\00\00\00\00\00", 
    [2 x i8] zeroinitializer, 
    i32 0, 
    i32 0, 
    %struct.tty_struct* null, 
    i32 0, 
    %struct.sem_undo* null, 
    %struct.sem_queue* null, 
    [8 x i8] zeroinitializer, 
    <{ i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, { %struct.i387_fsave_struct, [400 x i8] }, %struct.vm86_struct*, i32, i32, i32, i32, i32, [33 x i32], [1 x i32] }> <{
        i32 0, 
        i32 0, 
        i32 0, 
        i32 0, 
        i32 0, 
        [8 x i32] zeroinitializer, 
        i32 0, 
        i32 0, 
        i32 0, 
        { %struct.i387_fsave_struct, [400 x i8] } zeroinitializer, 
        %struct.vm86_struct* null, 
        i32 0, 
        i32 0, 
        i32 0, 
        i32 0, 
        i32 0, 
        [33 x i32] [ i32 -1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0 ], 
        [1 x i32] zeroinitializer }>, 
    %struct.fs_struct* @init_fs, 
    %struct.files_struct* @init_files, 
    %struct.namespace* null, 
    %struct.reiserfs_proc_info_data_t zeroinitializer, 
    %struct.signal_struct* @init_signals, 
    %struct.__kernel_fsid_t zeroinitializer, 
    %struct.sigpending {
        %struct.sigqueue* null, 
        %struct.sigqueue** getelementptr (%struct.task_union* bitcast (<{ <{ i32, i32, i32, %struct.atomic_t, %struct.exec_domain*, i32, i32, i32, i32, i32, i32, %struct.mm_struct*, i32, i32, i32, %struct.list_head, i32, %struct.task_struct*, %struct.task_struct*, %struct.mm_struct*, %struct.list_head, i32, i32, %struct.linux_binfmt*, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.list_head, %struct.task_struct*, %struct.task_struct**, %struct.wait_queue_head_t, %struct.completion*, i32, i32, i32, i32, i32, i32, i32, %struct.timer_list, %struct.fown_struct, i32, [1 x i32], [1 x i32], i32, i32, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, [32 x i32], i32, i32, i32, i8, [3 x i8], %struct.user_struct*, [11 x %struct.affs_bm_info], i16, [16 x i8], [2 x i8], i32, i32, %struct.tty_struct*, i32, %struct.sem_undo*, %struct.sem_queue*, [8 x i8], <{ i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, { %struct.i387_fsave_struct, [400 x i8] }, %struct.vm86_struct*, i32, i32, i32, i32, i32, [33 x i32], [1 x i32] }>, %struct.fs_struct*, %struct.files_struct*, %struct.namespace*, %struct.reiserfs_proc_info_data_t, %struct.signal_struct*, %struct.__kernel_fsid_t, %struct.sigpending, i32, i32, i32 (i8*)*, i8*, %struct.__kernel_fsid_t*, i32, i32, %struct.reiserfs_proc_info_data_t, i8*, %struct.llva_sigcontext*, i32, %struct.task_struct*, i32, %struct.llva_icontext_t, %struct.llva_fp_state_t, i32*, i32, i8*, [4 x i8*], [2 x i32] }>, [1632 x i32] }>* @init_task_union to %struct.task_union*), i32 0, i32 0, i32 100, i32 0), 
        %struct.__kernel_fsid_t zeroinitializer }, 
    i32 0, 
    i32 0, 
    i32 (i8*)* null, 
    i8* null, 
    %struct.__kernel_fsid_t* null, 
    i32 0, 
    i32 0, 
    %struct.reiserfs_proc_info_data_t zeroinitializer, 
    i8* null, 
    %struct.llva_sigcontext* null, 
    i32 0, 
    %struct.task_struct* null, 
    i32 0, 
    %struct.llva_icontext_t zeroinitializer, 
    %struct.llva_fp_state_t zeroinitializer, 
    i32* null, 
    i32 0, 
    i8* null, 
    [4 x i8*] zeroinitializer, 
    [2 x i32] zeroinitializer }>, [1632 x i32] zeroinitializer }>, section ".data.init_task", align 32		; <<{ <{ i32, i32, i32, %struct.atomic_t, %struct.exec_domain*, i32, i32, i32, i32, i32, i32, %struct.mm_struct*, i32, i32, i32, %struct.list_head, i32, %struct.task_struct*, %struct.task_struct*, %struct.mm_struct*, %struct.list_head, i32, i32, %struct.linux_binfmt*, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.list_head, %struct.task_struct*, %struct.task_struct**, %struct.wait_queue_head_t, %struct.completion*, i32, i32, i32, i32, i32, i32, i32, %struct.timer_list, %struct.fown_struct, i32, [1 x i32], [1 x i32], i32, i32, i32, i32, i32, i32, i8, [3 x i8], i32, i32, i32, i32, i32, i32, i32, i32, i32, [32 x i32], i32, i32, i32, i8, [3 x i8], %struct.user_struct*, [11 x %struct.affs_bm_info], i16, [16 x i8], [2 x i8], i32, i32, %struct.tty_struct*, i32, %struct.sem_undo*, %struct.sem_queue*, [8 x i8], <{ i32, i32, i32, i32, i32, [8 x i32], i32, i32, i32, { %struct.i387_fsave_struct, [400 x i8] }, %struct.vm86_struct*, i32, i32, i32, i32, i32, [33 x i32], [1 x i32] }>, %struct.fs_struct*, %struct.files_struct*, %struct.namespace*, %struct.reiserfs_proc_info_data_t, %struct.signal_struct*, %struct.__kernel_fsid_t, %struct.sigpending, i32, i32, i32 (i8*)*, i8*, %struct.__kernel_fsid_t*, i32, i32, %struct.reiserfs_proc_info_data_t, i8*, %struct.llva_sigcontext*, i32, %struct.task_struct*, i32, %struct.llva_icontext_t, %struct.llva_fp_state_t, i32*, i32, i8*, [4 x i8*], [2 x i32] }>, [1632 x i32] }>*> [#uses=2]
@init_fs = internal global %struct.fs_struct {
    %struct.atomic_t { i32 1 }, 
    %struct.reiserfs_proc_info_data_t zeroinitializer, 
    i32 18, 
    %struct.dentry* null, 
    %struct.dentry* null, 
    %struct.dentry* null, 
    %struct.vfsmount* null, 
    %struct.vfsmount* null, 
    %struct.vfsmount* null }, align 32		; <%struct.fs_struct*> [#uses=1]
@default_exec_domain = external global %struct.exec_domain		; <%struct.exec_domain*> [#uses=1]
@root_user = external global %struct.user_struct		; <%struct.user_struct*> [#uses=1]
@init_files = internal global %struct.files_struct {
    %struct.atomic_t { i32 1 }, 
    %struct.reiserfs_proc_info_data_t zeroinitializer, 
    i32 32, 
    i32 1024, 
    i32 0, 
    %struct.file** getelementptr (%struct.files_struct* @init_files, i32 0, i32 10, i32 0), 
    %struct.fd_set* getelementptr (%struct.files_struct* @init_files, i32 0, i32 8), 
    %struct.fd_set* getelementptr (%struct.files_struct* @init_files, i32 0, i32 9), 
    %struct.fd_set zeroinitializer, 
    %struct.fd_set zeroinitializer, 
    [32 x %struct.file*] zeroinitializer }, align 32		; <%struct.files_struct*> [#uses=4]
@init_signals = internal global %struct.signal_struct {
    %struct.atomic_t { i32 1 }, 
    [64 x %struct.k_sigaction] zeroinitializer, 
    %struct.reiserfs_proc_info_data_t zeroinitializer }, align 32		; <%struct.signal_struct*> [#uses=1]
@swapper_pg_dir = external global [1024 x %struct.atomic_t]		; <[1024 x %struct.atomic_t]*> [#uses=1]

declare void @it_real_fn(i32)
