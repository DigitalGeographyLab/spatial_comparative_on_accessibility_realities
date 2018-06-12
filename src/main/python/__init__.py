import getopt
import sys
import traceback

import geopandas as gpd

sys.path.append('/dgl/codes/spatial_comparative_on_accessibility_realities/')

from src.main.python.edu.accessibility.pt_routing.OpenTripPlanerRouterAccess import OpenTripPlanerRouterAccess
from src.main.python.edu.accessibility.util.utilitaries import FileActions, WGS_84, Logger


def run(originsFilename, destinationsFilename, outputFolder):
    Logger.getInstance().info("Origins: %s" % originsFilename)
    Logger.getInstance().info("Destination: %s" % destinationsFilename)
    Logger.getInstance().info("Output: %s" % outputFolder)

    fileActions = FileActions()
    routerAnalyst = OpenTripPlanerRouterAccess()

    fileActions.createFolder(folderPath=outputFolder)

    origins = gpd.read_file(originsFilename)
    destinations = gpd.read_file(destinationsFilename)
    origins = origins.to_crs(WGS_84)
    destinations = destinations.to_crs(WGS_84)

    routerAnalyst.processPlans(
        originPointsDF=origins,
        destinationsPointsDF=destinations,
        time='7:00am',
        date='05-07-2018',
        worstTime='8:00am'
    )


if __name__ == "__main__":
    try:
        argv = sys.argv[1:]
        opts, args = getopt.getopt(
            argv, "s:e:o:",
            ["origin_points=", "destination_points=", "output_folder="]
        )

        originsFilename = None
        destinationsFilename = None
        outputFolder = None

        for opt, arg in opts:
            if opt in "--help":
                # printHelp()
                break

            # print("options: %s, arg: %s" % (opt, arg))

            if opt in ("-s", "--origin_points"):
                originsFilename = arg

            if opt in ("-e", "--destination_points"):
                destinationsFilename = arg

            if opt in ("-o", "--output_folder"):
                outputFolder = arg

        Logger.configureLogger(outputFolder=outputFolder,
                               prefix="OpenTripPlanerRouterAccess")

        run(originsFilename=originsFilename,
            destinationsFilename=destinationsFilename,
            outputFolder=outputFolder)

    except Exception as err:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        Logger.getInstance().exception(''.join('>> ' + line for line in lines))
