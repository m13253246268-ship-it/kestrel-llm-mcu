cmake_minimum_required(VERSION 3.16)

# ESP-IDF 标准项目入口：依赖 IDF_PATH 环境变量（export.ps1 已设置）。
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

project(kestrel_mcu)
