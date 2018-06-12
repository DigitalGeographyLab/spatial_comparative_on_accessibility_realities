import os
import unittest

# from pandas.util.testing import assert_frame_equal
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc
from pandas.util.testing import assert_frame_equal

from src.main.python import run

plt.switch_backend('agg')
rc("text", usetex=False)

from src.main.python.edu.accessibility.pt_routing.OpenTripPlanerRouterAccess import OpenTripPlanerRouterAccess, \
    createEmptyTravelTimeDataFrame
from src.main.python.edu.accessibility.util.utilitaries import FileActions, getTimestampFromString, \
    Logger, WGS_84


class OpenTripPlanerRouterAccessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Logger.configureLogger(outputFolder=os.path.join(os.getcwd(), "output", ),
                               prefix="OpenTripPlanerRouterAccessTest")

    def setUp(self):
        self.fileActions = FileActions()
        self.maxDiff = None
        self.routerAnalyst = OpenTripPlanerRouterAccess()

    def test_givenAnOriginDestinationPoints_then_retrievePlanRoutes(self):
        originPointsURL = os.path.join(os.getcwd(), "src", "test", "resources", "input", "westPoint.geojson")
        destinationPointsURL = os.path.join(os.getcwd(), "src", "test", "resources", "input",
                                            "vuolukiventiePoint.geojson")
        expectedPlanURL = os.path.join(os.getcwd(), "src", "test", "resources", "output",
                                       "plan_5973737_to_5973738.json")

        outputFolder = os.path.join(os.getcwd(), "output", "plans")
        self.fileActions.createFolder(folderPath=outputFolder)

        origins = gpd.read_file(originPointsURL)
        destinations = gpd.read_file(destinationPointsURL)
        origins = origins.to_crs(WGS_84)
        destinations = destinations.to_crs(WGS_84)

        expected = self.fileActions.readJson(expectedPlanURL)

        for index, origin in origins.iterrows():
            for index, destination in destinations.iterrows():
                result = self.routerAnalyst.getRoutePlan(origin=origin.geometry,
                                                         destination=destination.geometry,
                                                         time='7:00am',
                                                         date='05-07-2018',
                                                         worstTime='8:00am')
                planFilename = "plan_%s_to_%s.json" % (origin["YKR_ID"], destination["YKR_ID"])
                self.fileActions.writeFile(folderPath=outputFolder,
                                           filename=planFilename,
                                           data=result)

                result = self.fileActions.readJson(os.path.join(outputFolder, planFilename))

                self.assertEqual(expected, result)

    def test_givenARoutePlan_then_extractPlanStats(self):
        planURL = os.path.join(os.getcwd(), "src", "test", "resources", "output",
                               "plan_West Point_to_Vuolukiventie.json")
        plan = self.fileActions.readJson(planURL)
        self.rushHourStartTime = getTimestampFromString("2018-05-07 07:00:00")
        self.rushHourEndTime = getTimestampFromString("2018-05-07 10:00:00")
        self.rushHourLastDepartureTime = getTimestampFromString("2018-05-07 09:00:00")
        duration_divider = 60  # seconds

        ptDataframe = pd.DataFrame(columns=[
            "from_id",
            "to_id",
            "wait_t",
            "walk_t",
            "walk_d",
            "transit_t",
            "travel_t",
            "min_b",  # min boardings
            "max_b",  # max boardings
            "avg_b",  # average boardings
            "fast_t",  # fastest route time
            "fast_r_b",  # fastest route boardings
            "max_b_t",  # max boardings travel time
            "po_t",  # pareto-optimal travel time
            "po_d",  # pareto-optimal travel distance
            "po_b",  # pareto-optimal boardings
            "avg_t",  # average travel time
            "1_boarding_probability_density",
            "2_boarding_probability_density",
            "3_boarding_probability_density"
        ])

        expectedFastestRoute = createEmptyTravelTimeDataFrame()
        expectedFastestRoute.loc[0] = [
            0,  # planIndex
            5,  # waitingTime
            12,  # walkTime
            22,  # transitTime
            40,  # duration
            1,  # transfers
            1525665854,  # startTime
            1525668267,  # endTime
            1,  # minBoardings
            2,  # maxBoardings
            1.3125,  # meanBoardings
            58,  # maxDuration
            48.375,  # meanDuration
            52.2,  # meanDurationMaxBoardings
            5.097487  # durationStandardDeviation
        ]

        fastestRoute = self.routerAnalyst.getFastestRoute(plan)

        print(fastestRoute)
        assert_frame_equal(expectedFastestRoute, fastestRoute)

    @unittest.SkipTest
    def test_givenASetOfOriginAndDestinationPoints_then_processPlans(self):
        originPointsURL = os.path.join(os.getcwd(), "src", "test", "resources", "input", "westPoint.geojson")
        destinationPointsURL = os.path.join(os.getcwd(), "src", "test", "resources", "input",
                                            "vuolukiventiePoint.geojson")

        outputFolder = os.path.join(os.getcwd(), "output", "plans")
        self.fileActions.createFolder(folderPath=outputFolder)

        origins = gpd.read_file(originPointsURL)
        destinations = gpd.read_file(destinationPointsURL)
        origins = origins.to_crs(WGS_84)
        destinations = destinations.to_crs(WGS_84)

        self.routerAnalyst.processPlans(
            originPointsDF=origins,
            destinationsPointsDF=destinations,
            time='7:00am',
            date='05-07-2018',
            worstTime='8:00am'
        )

    # @unittest.SkipTest
    def test_run(self):
        originPointsURL = os.path.join(os.getcwd(), "src", "test", "resources", "input", "tallin_500_grid.geojson")
        destinationPointsURL = os.path.join(os.getcwd(), "src", "test", "resources", "input",
                                            "tallin_500_grid.geojson")
        outputFolder = os.path.join(os.getcwd(), "output", "plans")
        run(originsFilename=originPointsURL,
            destinationsFilename=destinationPointsURL,
            outputFolder=outputFolder)

    @unittest.SkipTest
    def test_add_ids_to_grid(self):
        gridURL = os.path.join(os.getcwd(), "src", "test", "resources", "input", "tallin_500_grid.geojson")
        outputFolder = os.path.join(os.getcwd(), "output")
        origins = gpd.read_file(gridURL)
        origins["ykr_id"] = [x for x in range(0, len(origins), 1)]
        origins.to_file(os.path.join(outputFolder, "tallin_500_grid_ids.geojson"), driver='GeoJSON')
