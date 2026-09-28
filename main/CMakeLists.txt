idf_component_register(
    SRCS "main.c" "kmcu.c" "pie_dotprod.S"
    INCLUDE_DIRS "."
    REQUIRES esp_partition esp_timer esp_psram fatfs esp_driver_sdmmc sdmmc esp_driver_gpio esp_driver_uart
)
