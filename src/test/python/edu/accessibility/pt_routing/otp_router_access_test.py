# import unittest
#
#
# import sys
# import os
#
# import geopandas as gpd
# from shapely.geometry import Point
#
# from org.opentripplanner.scripting.api import OtpsEntryPoint
#
#
# WGS_84 = {'init': 'EPSG:4326'}
# PSEUDO_MERCATOR = {'init': 'EPSG:3857'}
#
#
# # class Point:
# #     def __init__(self, latitute, longitude, epsgCode):
# #         """
# #         Defines a point with latitute and longitude in a specific coordinate reference system.
# #
# #         :param latitute: latitude.
# #         :param longitude: longitude.
# #         :param epsgCode: Coordinate Reference System code.
# #         """
# #         self.__latitude = latitute
# #         self.__longitude = longitude
# #         self.__epsgCode = epsgCode
# #
# #     def getLatitude(self):
# #         return self.__latitude
# #
# #     def getLongitude(self):
# #         return self.__longitude
# #
# #     def getEPSGCode(self):
# #         return self.__epsgCode
# #
# #     def setLatitude(self, latitute):
# #         self.__latitude = latitute
# #
# #     def setLongitude(self, longitude):
# #         self.__longitude = longitude
# #
# #     def setEPSGCode(self, epsgCode):
# #         self.__epsgCode = epsgCode
# #
# #     def equals(self, endPoint):
# #         if not endPoint:
# #             return False
# #         return self.getLongitude() == endPoint.getLongitude() and self.getLatitude() == endPoint.getLatitude() and self.getEPSGCode() == endPoint.getEPSGCode()
#
#
# class OpenTripPlanerRouterAccessTest(unittest.TestCase):
#     def givenTwoPoints_then_retrievePossibleRoutes(self):
#         pathToDir = os.path.join(os.getcwd(), "src", "main", "resources")
#         subgraph_name = "Graph.obj"
#         otp = OtpsEntryPoint.fromArgs(["--graphs", pathToDir, "--router", subgraph_name])
#         originPointsURL = os.path.join(os.getcwd(), "test", "resources", "geojson", "northPoint.geojson")
#         destinationPointsURL = os.path.join(os.getcwd(), "test", "resources", "geojson", "rautatientoriPoint.geojson")
#         origins = gpd.read_file(originPointsURL)
#         origins.to_crs(WGS_84)
#         destinations = gpd.read_file(destinationPointsURL)
#         destinations.to_crs(WGS_84)
#         # origin = [float(43.757937), float(-79.315366)] # x = 381125, y = 6690630
#         # destination = [float(43.651911), float(-79.382175)]
#
#         router = otp.getRouter()
#
#         for index, origin in origins.iterrows():
#             # print row['c1'], row['c2']
#             for index, destination in destinations.iterrows():
#                 # print row['c1'], row['c2']
#
#                 r = otp.createRequest()
#                 year = 2018
#                 month = 7
#                 day = 18
#                 hour = 07
#                 minute = 00
#                 second = 00
#
#                 r.setDateTime(year, month, day, hour, minute, second)
#                 r.setModes('TRANSIT, WALK')
#                 r.setMaxTimeSec(4200)
#                 r.setOrigin(origin.x(), origins.y())
#                 r.setClampInitialWait(0)
#                 spt = router.plan(r)
#                 result = spt.eval(destination.x(), destination.y())
#                 self.assertEqual("", result)



####################JYTHON
# def givenTwoPoints_then_retrievePossibleRoutes(self):
#     pathToDir = os.path.join(os.getcwd(), "src", "main", "resources")
#     subgraph_name = "Graph.obj"
#     otp = OtpsEntryPoint.fromArgs(["--graphs", pathToDir, "--router", subgraph_name])
#     originPointsURL = os.path.join(os.getcwd(), "test", "resources", "geojson", "northPoint.geojson")
#     destinationPointsURL = os.path.join(os.getcwd(), "test", "resources", "geojson", "rautatientoriPoint.geojson")
#     origins = gpd.read_file(originPointsURL)
#     origins.to_crs(WGS_84)
#     destinations = gpd.read_file(destinationPointsURL)
#     destinations.to_crs(WGS_84)
#     # origin = [float(43.757937), float(-79.315366)] # x = 381125, y = 6690630
#     # destination = [float(43.651911), float(-79.382175)]
#
#     router = otp.getRouter()
#
#     for index, origin in origins.iterrows():
#         # print row['c1'], row['c2']
#         for index, destination in destinations.iterrows():
#             # print row['c1'], row['c2']
#
#             r = otp.createRequest()
#             year = 2018
#             month = 7
#             day = 18
#             hour = 07
#             minute = 00
#             second = 00
#
#             r.setDateTime(year, month, day, hour, minute, second)
#             r.setModes('TRANSIT, WALK')
#             r.setMaxTimeSec(4200)
#             r.setOrigin(origin.x(), origins.y())
#             r.setClampInitialWait(0)
#             spt = router.plan(r)
#             result = spt.eval(destination.x(), destination.y())
#             self.assertEqual("", result)


###########PARETO OPTIMAL
def test_givenARoutePlan_then_extractPlanStats_temp(self):
    planURL = os.path.join(os.getcwd(), "src", "test", "resources", "output",
                           "plan_West Point_to_Vuolukiventie.json")
    plan = self.fileActions.readJson(planURL)
    rushHourStartTime = getTimestampFromString("2018-05-07 07:00:00")
    rushHourEndTime = getTimestampFromString("2018-05-07 10:00:00")

    rushHourLastDepartureTime = getTimestampFromString("2018-05-07 09:00:00")

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
    # ptDataframe.min_b = ptDataframe.min_b.astype(np.int64)
    # ptDataframe.max_b = ptDataframe.max_b.astype(np.int64)
    # ptDataframe.mean_b = ptDataframe.mean_b.astype(np.int64)
    # ptDataframe.mean_b_2 = ptDataframe.mean_b_2.astype(np.int64)

    # ptDataframe.loc[len(ptDataframe)] = [min_n, max_n, mean_n, mean_temp_n, Point(lon, lat)]

    itineraryDF = self.routerAnalyst.createEmptyTravelTimeDataFrame()

    # itineraryDF.waitingTime = itineraryDF.waitingTime.astype(np.int64)
    # itineraryDF.walkTime = itineraryDF.walkTime.astype(np.int64)
    # itineraryDF.transitTime = itineraryDF.transitTime.astype(np.int64)
    # itineraryDF.duration = itineraryDF.duration.astype(np.int64)
    # itineraryDF.transfers = itineraryDF.transfers.astype(np.int64)
    # itineraryDF.departureTime = itineraryDF.departureTime.astype(np.int64)
    # itineraryDF.arrivalTime = itineraryDF.arrivalTime.astype(np.int64)

    count = 0
    for itinerary in plan["plan"]["itineraries"]:
        if itinerary["endTime"] / 1000 < rushHourEndTime:
            itineraryDF.loc[len(itineraryDF)] = [
                count,
                int(itinerary["waitingTime"] / 60),
                int(itinerary["walkTime"] / 60),
                int(itinerary["transitTime"] / 60),
                # int(itinerary["duration"] / 60),
                itinerary["duration"],
                itinerary["transfers"],
                int(itinerary["startTime"] / 1000),
                int(itinerary["endTime"] / 1000)
            ]
        count += 1

    itineraryDF = itineraryDF.sort_values(by=['startTime', 'endTime'], ascending=True)

    fastestRoutes = itineraryDF[
        (rushHourStartTime < itineraryDF["startTime"]) &
        (itineraryDF["startTime"] <= rushHourLastDepartureTime)
        ]
    minArrivalTime = min(itineraryDF["endTime"])
    lastFastRoute = itineraryDF[
        (rushHourLastDepartureTime < itineraryDF["startTime"]) &
        (itineraryDF["endTime"] <= minArrivalTime)
        ]

    lastFastRoute = lastFastRoute.sort_values(by=['startTime', 'endTime'], ascending=True)

    fastestRoutes = fastestRoutes.append(lastFastRoute.head(1), ignore_index=True)

    # paretoFrontier = pd.DataFrame(columns=[
    #     "planIndex",
    #     "waitingTime",
    #     "walkTime",
    #     "transitTime",
    #     "duration",
    #     "transfers",
    #     "startTime",
    #     "endTime"
    # ])
    #
    # #### pareto frontier
    # # dominates = (
    # #     self.arrival_time_target <= other.arrival_time_target and
    # #     self.first_leg_is_walk <= other.first_leg_is_walk
    # # )
    #
    # currentBestParetoFrontier = paretoFrontier.copy()
    #
    # for index, routeRow in fastestRoutes.iterrows():
    #     is_dominated = False
    #     isWalkMode_routeRow = plan["plan"]["itineraries"][routeRow["planIndex"]]["legs"][0]["mode"] == "WALK"
    #
    #     for paretoIndex, paretoRow in currentBestParetoFrontier.iterrows():
    #         isWalkMode_paretoRow = plan["plan"]["itineraries"][paretoRow["planIndex"]]["legs"][0]["mode"] == "WALK"
    #         if (paretoRow["endTime"] <= routeRow["endTime"]) and \
    #                 (isWalkMode_paretoRow <= isWalkMode_routeRow):
    #             is_dominated = True
    #         break
    #     if not is_dominated:
    #         paretoFrontier = paretoFrontier.append(routeRow)
    #         new_best = pd.DataFrame(columns=[
    #             "planIndex",
    #             "waitingTime",
    #             "walkTime",
    #             "transitTime",
    #             "duration",
    #             "transfers",
    #             "startTime",
    #             "endTime"
    #         ])
    #         for paretoIndex, oldParetoRow in currentBestParetoFrontier.iterrows():
    #             isWalkMode_oldParetoRow = plan["plan"]["itineraries"][oldParetoRow["planIndex"]]["legs"][0]["mode"] == "WALK"
    #             if not ((routeRow["endTime"] <= oldParetoRow["endTime"]) and \
    #                     (isWalkMode_routeRow <= isWalkMode_oldParetoRow)):
    #                 new_best = new_best.append(oldParetoRow)
    #         new_best = new_best.append(routeRow)
    #         currentBestParetoFrontier = new_best
    # ####



    itineraryGrouped = fastestRoutes.groupby('transfers')
    minBoardings = min(fastestRoutes["transfers"])
    maxBoardings = max(fastestRoutes["transfers"])
    duration_divider = 60  # seconds
    # n_boardings_to_fill_color, n_boardings_to_line_color = self._get_fill_and_line_colors(
    #     minBoardings, maxBoardings
    # )
    n_boardings_range = range(minBoardings, maxBoardings + 1)
    default_lw = 5
    # n_boardings_to_lw = {n: default_lw for i, n in enumerate(n_boardings_range)}

    fig = plt.figure(figsize=(11, 3.5))
    subplot_grid = (1, 8)
    ax1 = plt.subplot2grid(subplot_grid, (0, 0), colspan=6, rowspan=1)
    # x_axis_formatter = md.DateFormatter("%H:%M")
    # ax1.xaxis.set_major_formatter(x_axis_formatter)
    # ax1.set_xlim(
    #     rushHourStartTime,
    #     rushHourEndTime
    # )

    walking_is_fastest_time = 0
    non_walk_blocks = []

    tdist_split_points = set()
    previous_dep_time = rushHourStartTime
    minWalkTime = min(fastestRoutes["walkTime"])

    for index, routeRow in fastestRoutes.iterrows():

        # lw = n_boardings_to_lw[transfers]
        # xs = [_ut_to_unloc_datetime(prev_dep_time), _ut_to_unloc_datetime(journey_label.departure_time)]
        # ys = numpy.array([journey_label.duration() + waiting_time, journey_label.duration()]) / duration_divider]
        #
        # ax1.plot(xs, ys,
        #          color=n_boardings_to_line_color[transfers],  # "k",
        #          lw=lw)
        # ax1.fill_between(xs, 0, ys, color=n_boardings_to_fill_color[transfers])

        if previous_dep_time >= rushHourEndTime:
            break
        departureTime = routeRow["startTime"]
        arrivalTime = routeRow["endTime"]
        duration = routeRow["duration"]

        end_time = min(departureTime, rushHourEndTime)

        if routeRow["transitTime"] == 0:  # test for walk
            walking_is_fastest_time += arrivalTime - departureTime
        else:

            temporalDistanceToStart = duration + (departureTime - previous_dep_time)
            # toAdd = False
            # if temporalDistanceToStart > minWalkTime:
            #     split_point_x_computed = departureTime - (minWalkTime - duration)
            #     split_point_x = min(split_point_x_computed, end_time)
            #     if previous_dep_time < split_point_x:
            #         # add walk block, only if it is required
            #
            #         temporalDistanceToStart = minWalkTime
            #         temporalDistanceToEnd = minWalkTime
            #
            #         toAdd = True
            #
            #     if split_point_x < end_time:
            #         temporalDistanceToStart = duration + (end_time - split_point_x)
            #         temporalDistanceToEnd = temporalDistanceToStart - (end_time - previous_dep_time)
            #         toAdd = True
            # else:
            #     if previous_dep_time < end_time:
            #         temporalDistanceToStart = temporalDistanceToStart
            #         temporalDistanceToEnd = temporalDistanceToStart - (end_time - previous_dep_time)
            #         toAdd = True
            #
            # if toAdd:
            #     non_walk_blocks.append(routeRow)
            #     tdist_split_points.add(temporalDistanceToEnd)
            #     tdist_split_points.add(temporalDistanceToStart)

            if previous_dep_time < end_time:
                temporalDistanceToStart = duration + (departureTime - previous_dep_time)
                temporalDistanceToEnd = temporalDistanceToStart - (end_time - previous_dep_time)
                routeRow["temporalDistanceToStart"] = temporalDistanceToStart
                routeRow["temporalDistanceToEnd"] = temporalDistanceToEnd

                non_walk_blocks.append(routeRow)
                tdist_split_points.add(temporalDistanceToEnd)
                tdist_split_points.add(temporalDistanceToStart)

                previous_dep_time = arrivalTime

    distance_split_points_ordered = np.array(sorted(list(tdist_split_points)))
    temporal_distance_split_widths = (distance_split_points_ordered[1:] - distance_split_points_ordered[
                                                                          :-1]) / duration_divider

    fill_colors, line_colors = self._get_fill_and_line_colors(minBoardings, maxBoardings)

    temporal_distance_values_to_plot = []
    for x in distance_split_points_ordered:
        temporal_distance_values_to_plot.append(x)
        temporal_distance_values_to_plot.append(x)
    temporal_distance_values_to_plot = np.array(temporal_distance_values_to_plot)

    pdf_values_to_plot_by_n_boardings = {}
    pdf_areas = {}

    for n_boardings in range(minBoardings, maxBoardings + 1):
        # if walking, the block has no "n_boardings" attribute
        blocks_now = [block for block in non_walk_blocks if block["transfers"] >= n_boardings]

        journey_counts = np.zeros(len(temporal_distance_split_widths))
        for block_now in blocks_now:
            first_index = np.searchsorted(distance_split_points_ordered, block_now["temporalDistanceToEnd"])
            last_index = np.searchsorted(distance_split_points_ordered, block_now["temporalDistanceToStart"])
            journey_counts[first_index:last_index] += 1

        part_pdf = journey_counts / (rushHourEndTime - rushHourStartTime)
        pdf_areas[n_boardings] = sum(part_pdf * temporal_distance_split_widths)

        pdf_values_to_plot = [0]
        for pdf_value in part_pdf:
            pdf_values_to_plot.append(pdf_value)
            pdf_values_to_plot.append(pdf_value)
        pdf_values_to_plot.append(0)
        pdf_values_to_plot_by_n_boardings[n_boardings] = np.array(pdf_values_to_plot)

    for n_boardings in range(max(1, minBoardings), maxBoardings + 1):
        if n_boardings is maxBoardings:
            prob = pdf_areas[n_boardings]
        else:
            prob = pdf_areas[n_boardings] - pdf_areas[n_boardings + 1]
        pdf_values_to_plot = pdf_values_to_plot_by_n_boardings[n_boardings]
        if n_boardings is 0:
            label = "$P(\\mathrm{walk})= %.2f $" % (prob)
        else:
            label = "$P(b=" + str(n_boardings) + ")= %.2f $" % (prob)
            # "$P_\\mathrm{" + self.n_boardings_to_label(n_boardings).replace(" ", "\\,") + "} = %.2f $" % (prob)
        ax1.fill_betweenx(temporal_distance_values_to_plot / duration_divider,
                          pdf_values_to_plot * duration_divider,
                          label=label,
                          color=fill_colors[n_boardings], zorder=n_boardings)
        ax1.plot(pdf_values_to_plot * duration_divider,
                 temporal_distance_values_to_plot / duration_divider,
                 color=line_colors[n_boardings], zorder=n_boardings)

    legend_font_size = 12
    legend_loc = "best"
    leg = ax1.legend(loc=legend_loc, fancybox=True, prop={"size": legend_font_size})
    leg.get_frame().set_alpha(0.9)
    # return ax1
    fig.tight_layout()
    outputFolder = os.path.join(os.getcwd(), "output", "plans")
    fig.savefig(os.path.join(outputFolder, "probability_.pdf"))

def _get_fill_and_line_colors(self, min_n, max_n):
    colorPerBoarding = self._get_colors_for_boardings(min_n, max_n)
    n_boardings_range = range(min_n, max_n + 1)
    nboardings_to_color = {n: colorPerBoarding[i] for i, n in enumerate(n_boardings_range)}

    n_boardings_to_line_color = {}
    n_boardings_to_fill_color = {}

    #
    rgbs = [color_tuple[:3] for color_tuple in nboardings_to_color.values()]
    hsvs = colors.rgb_to_hsv(rgbs)
    max_saturation = max([hsv[1] for hsv in hsvs])
    line_saturation_multiplier = 1 / max_saturation

    for n, color_tuple in nboardings_to_color.items():
        c = self._multiply_color_saturation(color_tuple, line_saturation_multiplier)
        c = self._multiply_color_brightness(c, 1)
        n_boardings_to_line_color[n] = c

        c = self._multiply_color_brightness(color_tuple, 1.2)
        c = self._multiply_color_saturation(c, 0.8)
        n_boardings_to_fill_color[n] = c
    return n_boardings_to_fill_color, n_boardings_to_line_color

def _get_colors_for_boardings(self, min_n_boardings, max_n_boardings):
    cmap = self.get_colormap_for_boardings(max_n_boardings)
    colors = [cmap(float(n_boardings) / max_n_boardings) if max_n_boardings is not 0 else cmap(0.0) for n_boardings
              in range(int(max_n_boardings) + 1)]
    return colors[min_n_boardings:max_n_boardings + 1]

def get_colormap_for_boardings(self, max_n_boardings=None):
    n_default = 5
    if max_n_boardings in [float('nan'), None]:
        max_n_boardings = n_default
    from matplotlib import cm
    cmap = cm.get_cmap("cubehelix_r")
    start = 0.1
    end = 0.9
    if max_n_boardings is 0:
        step = 0
    else:
        divider = max(n_default, max_n_boardings)
        step = (end - start) / divider
    truncated = self._truncate_colormap(cmap, start, start + step * max_n_boardings)
    return truncated

def _multiply_color_saturation(self, color, multiplier):
    hsv = colors.rgb_to_hsv(color[:3])
    rgb = colors.hsv_to_rgb((hsv[0], hsv[1] * multiplier, hsv[2]))
    return list(iter(rgb)) + [1]

@classmethod
def _multiply_color_brightness(self, color, multiplier):
    hsv = colors.rgb_to_hsv(color[:3])
    rgb = colors.hsv_to_rgb((hsv[0], hsv[1], max(0, min(1, hsv[2] * multiplier))))
    return list(iter(rgb)) + [1]

def _truncate_colormap(self, cmap, minval=0.0, maxval=1.0, n=100):
    """
    Truncates a colormap to use.
    Code originall from http://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap
