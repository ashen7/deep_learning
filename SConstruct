#encoding:utf-8
#python scons script file

current_dir = '/home/yipeng/workspace/deep_learning/'         #项目目录
third_party_library_dir = '/home/yipeng/third_party_library/' #第三方库目录

current_src = current_dir + 'src/'

current_inc = current_dir + 'include/'
glog_inc = third_party_library_dir + 'glog/include'
gflags_inc = third_party_library_dir + 'gflags/include'
ffmpeg_inc = third_party_library_dir + 'ffmpeg/include'
protobuf_inc = third_party_library_dir + 'protobuf/include'
opencv_inc = third_party_library_dir + 'opencv/include'

glog_lib = third_party_library_dir + 'glog/lib'
gflags_lib = third_party_library_dir + 'gflags/lib'
ffmpeg_lib = third_party_library_dir + 'ffmpeg/lib'
protobuf_lib = third_party_library_dir + 'protobuf/lib'
opencv_lib = third_party_library_dir + 'opencv/lib'

#cpp头文件路径
include_dirs = [
    current_inc, 
    glog_inc, 
    gflags_inc, 
    ffmpeg_inc, 
    protobuf_inc, 
    opencv_inc, 
]

#cpp库文件路径
lib_dirs = [
    glog_lib, 
    gflags_lib, 
    ffmpeg_lib, 
    protobuf_lib, 
    opencv_lib, 
]

#cpp库文件  动态链接库 或者静态库
libs = [
    'glog', 
    'gflags',
    'protobuf', 
    'opencv_core', 
    'opencv_highgui', 
    'opencv_imgproc', 
    'opencv_imgcodecs', 
    'opencv_videoio', 
    'opencv_objdetect', 
    'avcodec', 
    'avformat', 
    'avutil', 
    'swscale', 
]

#链接时的标志  -Wl指定运行可执行程序时 去哪个路径找动态链接库
link_flags = [
    '-pthread', 
    '-fsanitize=address', 
    '-Wl,-rpath-link=' + ":".join(lib_dirs), 
    '-Wl,-rpath=' + ":".join(lib_dirs), 
]

#cpp的编译标志
cpp_flags = [
    '-O3',                     #更好的编译优化
    '-fsanitize=address',      #asan的选项 编译链接都要用
    '-fno-omit-frame-pointer', #堆栈跟踪  
    '-g', 
    '-std=gnu++11', 
    '-W',                      #显示所有warning
    '-Wall', 
]

fcnn_source = [
    current_dir + 'test_fcnn.cpp', 
    current_src + 'full_connected_layer.cpp', 
    current_src + 'neural_network.cpp'
]

#program生成可执行文件
video_decode = Program(
    target = 'fcnn',              #可执行文件名   -o
    source = fcnn_source,         #源文件列表
    CPPPATH = include_dirs,       #头文件路径列表 -I
    LIBPATH = lib_dirs,           #库文件路径列表 -L
    LIBS = libs,                  #库文件  -l
    LINKFLAGS = link_flags,       #链接的标志  -
    CPPFLAGS = cpp_flags,         #编译的标志  -
)

#安装
#bin_path = current_dir
#Install(bin_path, "fcnn")
