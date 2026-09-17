import Metashape

import plugin_align_and_analyse
import plugin_import_geostix

if __name__ == "__main__":
    application: Metashape.Application = Metashape.app

    application.removeMenuItem(plugin_import_geostix.MENU_ITEM_NAME)
    application.addMenuItem(plugin_import_geostix.MENU_ITEM_NAME, plugin_import_geostix.import_geostix_tool)

    application.removeMenuItem(plugin_align_and_analyse.MENU_ITEM_NAME)
    application.addMenuItem(plugin_align_and_analyse.MENU_ITEM_NAME, plugin_align_and_analyse.align_and_analyse)
