project('TEdgeMask', 'cpp',
        version: '1',
        default_options: ['cpp_std=c++11', 'buildtype=release'],
        meson_version: '>=0.46')


warnings = [
  '-Wall',
  '-Wextra',
  '-Wshadow',
]

cflags = [
  warnings,
]

ldflags = [
]


host_cpu_family = host_machine.cpu_family()

if host_cpu_family.startswith('x86')
  cflags += ['-mfpmath=sse', '-msse2', '-DTEDGEMASK_X86=1']
endif


host_system = host_machine.system()

if host_system == 'windows' or host_system == 'cygwin'
  if host_cpu_family == 'x86'
    cflags += '-mstackrealign'
    ldflags += '-Wl,--kill-at'
  endif
endif


cxx = meson.get_compiler('cpp')


sources = [
  'src/tedgemask.cpp',
]

deps = [
  dependency('vapoursynth').partial_dependency(includes: true, compile_args: true),
  cxx.find_library('m', required: false),
]

shared_module('tedgemask',
              sources,
              dependencies: deps,
              link_args: ldflags,
              cpp_args: cflags,
              install: true)
