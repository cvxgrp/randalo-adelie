project(
  'example',
  'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++11',
  ],
)

py = import('python').find_installation(pure: false)
pybind11_dep = dependency('pybind11')
adelie_dep = dependency('adelie')
eigen3_dep = dependency('eigen3')
gomp_dep = dependency('gomp')

py.extension_module('example',
  'example.cpp',
  install: true,
  dependencies : [pybind11_dep, adelie_dep, eigen3_dep, gomp_dep],
)

