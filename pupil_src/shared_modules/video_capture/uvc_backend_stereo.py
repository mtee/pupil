'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import time
import logging
import uvc
import cv2
import numpy as np
from unicodedata import name

from version_utils import VersionFormat
from .base_backend import InitialisationError, Base_Source, Base_Manager

# check versions for our own depedencies as they are fast-changing
assert VersionFormat(uvc.__version__) >= VersionFormat('0.11')

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class UVCSourceStereo(Base_Source):
    """
    Camera Capture is a class that encapsualtes uvc.Capture:
    """

    stereoPairUID = ''

    # heuristic to find UIDs of the sensor stereo pair.
    # Works like that: get all UVC devices, then search for those that
    # a) contain PUPIL
    # AND
    # b) have the same name, e.g.
    def findPossibleUIDs(self, searchstring="Pupil"):
        seen = set()
        uniq = []
        namecandidate = ''
        left_uid = ''
        right_uid = ''
        found = 0
        self.stereoPairUID = None
        for d in self.devices:
            if d['name'] not in seen:
                if searchstring in d['name']:
                    uniq.append(d['name'])
                    seen.add(d['name'])
                    found = 1
            else:
                namecandidate = d['name']
        if found == 1:
            for d in self.devices:
                if left_uid == '' and d['name'] == namecandidate:
                    logger.debug("found match {}".format(namecandidate))
                    left_uid = d['uid']
                else:
                    logger.debug("no match")
                    if d['name'] == namecandidate:
                        right_uid = d['uid']
        logger.debug("left uid: {}".format(left_uid))
        logger.debug("right uid: {}".format(right_uid))
        return (left_uid, right_uid)

    def __init__(self, g_pool, frame_size, frame_rate, name=None, preferred_names=(), uid_left=None, uid_right=None,
                 uvc_controls={}):
        super().__init__(g_pool)
        self.uvc_capture_left = None
        self.uvc_capture_right = None
        logger.debug("__init__ of backend stereo called")
        if (preferred_names):
            logger.debug("preferred names: ")
            logger.debug(preferred_names)
        if (uid_left):
            logger.debug("uid left: " + uid_left)
        if (uid_right):
            logger.debug("uid right: " + uid_right)
        self._restart_in = 3
        assert name or preferred_names or uid_left
        self.devices = uvc.Device_List()

        # TODO: following line filters devices by the name,
        # which does not work with the stereo camera because in this case two cameras have the same name

        # ... we could try to choose the two preferred stereo cameras that have the same name
        devices_by_name = {dev['name']: dev for dev in self.devices}

        logger.debug("all devices:{}".format(self.devices))

        logger.debug("devices_by_name:{}".format(devices_by_name))

        # if uid is supplied we init with that
        if uid_left and uid_right:
            logger.debug("UIDSs supplied for left and right sensor")
            try:
                self.uvc_capture_left = uvc.Capture(uid_left)
                self.uvc_capture_right = uvc.Capture(uid_right)
            except uvc.OpenError:
                logger.warning("No avalilable camera found that matched {}".format(preferred_names))
            except uvc.InitError:
                logger.error("Camera failed to initialize.")
            except uvc.DeviceNotFoundError:
                logger.warning("No camera found that matched {}".format(preferred_names))

        # otherwise we use name or preffered_names
        else:
            logger.debug("no UIDSs supplied, trying hard coded values: {}".format(name))
            if name:
                preferred_names = (name,)
            else:
                pass
            assert preferred_names

            # try to init by name
            left, right = self.findPossibleUIDs("Pupil")

            try:
                logger.debug("Trying to open device: {}".format(left))
                self.uvc_capture_left = uvc.Capture(left)

                logger.debug("Trying to open device: {}".format(right))
                self.uvc_capture_right = uvc.Capture(right)
            except uvc.OpenError:
                logger.info("{} and {} matches {} but is already in use or blocked.".format(left, right, name))
            except uvc.InitError:
                logger.error("Camera failed to initialize.")

        # check if we were sucessfull
        if not self.uvc_capture_left:
            logger.error("Left Init failed. Capture is started in ghost mode. No images will be supplied.")
            self.name_backup = preferred_names
            self.frame_size_backup = frame_size
            self.frame_rate_backup = frame_rate
            if not self.uvc_capture_right:
                logger.error("Right Init failed. Capture is started in ghost mode. No images will be supplied.")
                self.name_backup = preferred_names
                self.frame_size_backup = frame_size
                self.frame_rate_backup = frame_rate
        else:
            logger.debug("Both captures initialized")
            #     frame_size = [1280, 720]
            logger.debug("configuring left capture device {} {}".format(frame_size, frame_rate))
            self.configure_capture(frame_size, frame_rate, uvc_controls)

            #            self.uvc_capture_left.bandwidth_factor = 2.0
            #            self.uvc_capture_right.bandwidth_factor = 2.0

            self.name_backup = (self.name,)
            self.frame_size_backup = frame_size
            self.frame_rate_backup = frame_rate

    def configure_capture(self, _frame_size, _frame_rate, uvc_controls):
        # Set camera defaults. Override with previous settings afterwards
        # if 'C930e' in self.uvc_capture.name:
        #    logger.debug('Timestamp offset for c930 applied: -0.1sec')
        #    self.ts_offset = -0.1
        # else:
        self.ts_offset = 0.0

        # UVC setting quirks:
        controls_dict = dict([(c.display_name, c) for c in self.uvc_capture_left.controls])

        # self.frame_size = frame_size

        logger.debug("possible sizes left: {}".format(self.uvc_capture_left.frame_sizes))
        logger.debug("possible sizes right: {}".format(self.uvc_capture_right.frame_sizes))

        self.frame_size = _frame_size
        self.frame_rate = _frame_rate
        # self.uvc_capture_right.frame_rate(frame_rate)
        for c in self.uvc_capture_left.controls:
            try:
                c.value = uvc_controls[c.display_name]
            except KeyError:
                logger.debug('No UVC setting "{}" found from settings.'.format(c.display_name))

        try:
            controls_dict['Auto Focus'].value = 0
        except KeyError:
            pass

        if ("Pupil Cam1" in self.uvc_capture_left.name or
                    "USB2.0 Camera" in self.uvc_capture_left.name):

            if ("ID0" in self.uvc_capture_left.name or "ID1" in self.uvc_capture_left.name):

                self.uvc_capture_left.bandwidth_factor = 1.3
                self.uvc_capture_right.bandwidth_factor = 1.3

                try:
                    controls_dict['Auto Exposure Priority'].value = 0
                except KeyError:
                    pass

                try:
                    controls_dict['Auto Exposure Mode'].value = 1
                except KeyError:
                    pass

                try:
                    controls_dict['Saturation'].value = 0
                except KeyError:
                    pass

                try:
                    controls_dict['Absolute Exposure Time'].value = 63
                except KeyError:
                    pass

                try:
                    controls_dict['Backlight Compensation'].value = 2
                except KeyError:
                    pass

                try:
                    controls_dict['Gamma'].value = 100
                except KeyError:
                    pass

            else:
                self.uvc_capture_left.bandwidth_factor = 2.0
                self.uvc_capture_right.bandwidth_factor = 2.0

                try:
                    controls_dict['Auto Exposure Priority'].value = 1
                except KeyError:
                    pass
        else:
            self.uvc_capture_left.bandwidth_factor = 3.0
            self.uvc_capture_right.bandwidth_factor = 3.0
            try:
                controls_dict['Auto Focus'].value = 0
            except KeyError:
                pass

    def _re_init_capture(self, uid_left, uid_right):
        if self.uvc_capture_left:
            current_size = self.uvc_capture_left.frame_size
            current_fps = self.uvc_capture_left.frame_rate
            current_uvc_controls = self._get_uvc_controls()
            self.uvc_capture_left.close()
        if self.uvc_capture_right:
            current_size = self.uvc_capture_right.frame_size
            current_fps = self.uvc_capture_right.frame_rate
            current_uvc_controls = self._get_uvc_controls()
            self.uvc_capture_right.close()
        self.deinit_gui()

        self.uvc_capture_left = uvc.Capture(uid_left)
        self.uvc_capture_right = uvc.Capture(uid_right)
        self.configure_capture(current_size, current_fps, current_uvc_controls)
        self.init_gui()


    def _init_capture(self, uid_left, uid_right):
        self.deinit_gui()
        self.uvc_capture_left = uvc.Capture(uid_left)
        self.uvc_capture_right = uvc.Capture(uid_right)
        self.configure_capture(self.frame_size_backup, self.frame_rate_backup, self._get_uvc_controls())
        self.init_gui()


    def _re_init_capture_by_names(self, names):
        # burn-in test specific. Do not change text!
        self.devices.update()
        print(names)
        left, right = self.findPossibleUIDs("Pupil")
        if self.uvc_capture_left or self.uvc_capture_right:
            self._re_init_capture(left, right)
        else:
            self._init_capture(left, right)
        raise InitialisationError('Could not find Camera {} during re initilization.'.format(names))


    def _restart_logic(self):
        if self._restart_in <= 0:
            if self.uvc_capture_left or self.uvc_capture_right:
                logger.warning("Capture failed to provide frames. Attempting to reinit.")
                self.name_backup = (self.uvc_capture_left.name,)
                self.uvc_capture_left = None
            try:
                self._re_init_capture_by_names(self.name_backup)
            except (InitialisationError, uvc.InitError):
                time.sleep(0.02)
                self.deinit_gui()
                self.init_gui()
            self._restart_in = int(5 / 0.02)
        else:
            self._restart_in -= 1


    def recent_events(self, events):
        # Add second frame here
        try:
            #   logger.debug("bandwidth right: {}".format(self.uvc_capture_right.bandwidth_factor))
            #   logger.debug("bandwidth left: {}".format(self.uvc_capture_left.bandwidth_factor))
            frame_right = self.uvc_capture_right.get_frame(0.05)

            frame_left = self.uvc_capture_left.get_frame(0.001)

            frame_left.timestamp = self.g_pool.get_timestamp() + self.ts_offset
            frame_right.timestamp = self.g_pool.get_timestamp() + self.ts_offset
        except uvc.StreamError:
            self._recent_frame = None

            self._restart_logic()
        except (AttributeError, uvc.InitError):
            self._recent_frame = None
            time.sleep(0.02)
            self._restart_logic()
        else:
          #  self._recent_frame = frame_right
          #  events['frame'] = frame_right

            self._recent_frame = frame_right
            events['frame'] = frame_right

            events['frame_right'] = frame_left
            self._restart_in = 3


    def _get_uvc_controls(self):
        d = {}

        if self.uvc_capture_left:
            for c in self.uvc_capture_left.controls:
                d[c.display_name] = c.value
        return d


    def get_init_dict(self):
        d = super().get_init_dict()
        d['frame_size'] = self.frame_size
        d['frame_rate'] = self.frame_rate
        if self.uvc_capture_left:
            d['name'] = self.name
            d['uvc_controls'] = self._get_uvc_controls()
        else:
            d['preferred_names'] = self.name_backup
        return d


    @property
    def name(self):
        if self.uvc_capture_left:
            return self.uvc_capture_left.name
        else:
            return "Ghost capture"


    @property
    def frame_size(self):
        if self.uvc_capture_left:
            return self.uvc_capture_left.frame_size
        else:
            if self.uvc_capture_right:
                return self.uvc_capture_right.frame_size
            else:
                return self.frame_size_backup


    @frame_size.setter
    def frame_size(self, new_size):
        # closest match for size
        logger.debug("frame size setter called")
        sizes = [abs(r[0] - new_size[0]) for r in self.uvc_capture_left.frame_sizes]
        best_size_idx = sizes.index(min(sizes))
        size = self.uvc_capture_left.frame_sizes[best_size_idx]
        if tuple(size) != tuple(new_size):
            logger.warning("%s resolution capture mode not available. Selected {}.".format(new_size, size))
        self.uvc_capture_left.frame_size = size
        self.uvc_capture_right.frame_size = size
        self.frame_size_backup = size


    @property
    def frame_rate(self):
        if self.uvc_capture_left:
            return self.uvc_capture_left.frame_rate
        else:
            return self.frame_rate_backup


    @frame_rate.setter
    def frame_rate(self, new_rate):
        # closest match for rate
        rates = [abs(r - new_rate) for r in self.uvc_capture_left.frame_rates]
        best_rate_idx = rates.index(min(rates))
        rate = self.uvc_capture_left.frame_rates[best_rate_idx]
        if rate != new_rate:
            logger.warning("{}fps capture mode not available at ({}) on '{}'. Selected {}fps. ".format(
                new_rate, self.uvc_capture_left.frame_size, self.uvc_capture_left.name, rate))
        self.uvc_capture_left.frame_rate = rate
        self.uvc_capture_right.frame_rate = rate
        self.frame_rate_backup = rate

    @property
    def jpeg_support(self):
        return True


    @property
    def online(self):
        return bool(self.uvc_capture_left) and bool(self.uvc_capture_right)

    def init_gui(self):
        from pyglui import ui
        ui_elements = []

        # lets define some  helper functions:
        def gui_load_defaults():
            for c in self.uvc_capture_left.controls:
                try:
                    c.value = c.def_val
                except:
                    pass

        def gui_update_from_device():
            for c in self.uvc_capture_left.controls:
                c.refresh()

        def set_frame_size(new_size):
            self.frame_size = new_size

        if self.uvc_capture_left is None:
            ui_elements.append(ui.Info_Text('Capture initialization failed.'))
            self.g_pool.capture_source_menu.extend(ui_elements)
            return

        ui_elements.append(ui.Info_Text('{} Controls'.format(self.name)))
        sensor_control = ui.Growing_Menu(label='Sensor Settings')
        sensor_control.append(ui.Info_Text("Do not change these during calibration or recording!"))
        sensor_control.collapsed = False
        image_processing = ui.Growing_Menu(label='Image Post Processing')
        image_processing.collapsed = True

        sensor_control.append(ui.Selector(
            'frame_size', self,
            setter=set_frame_size,
            selection=self.uvc_capture_left.frame_sizes,
            label='Resolution'
        ))

        def frame_rate_getter():
            return (self.uvc_capture_left.frame_rates, [str(fr) for fr in self.uvc_capture_left.frame_rates])

        sensor_control.append(ui.Selector('frame_rate', self, selection_getter=frame_rate_getter, label='Frame rate'))

        for control in self.uvc_capture_left.controls:
            c = None
            ctl_name = control.display_name

            # now we add controls
            if control.d_type == bool:
                c = ui.Switch('value', control, label=ctl_name, on_val=control.max_val, off_val=control.min_val)
            elif control.d_type == int:
                c = ui.Slider('value', control, label=ctl_name, min=control.min_val, max=control.max_val,
                              step=control.step)
            elif type(control.d_type) == dict:
                selection = [value for name, value in control.d_type.items()]
                labels = [name for name, value in control.d_type.items()]
                c = ui.Selector('value', control, label=ctl_name, selection=selection, labels=labels)
            else:
                pass
            # if control['disabled']:
            #     c.read_only = True
            # if ctl_name == 'Exposure, Auto Priority':
            #     # the controll should always be off. we set it to 0 on init (see above)
            #     c.read_only = True

            if c is not None:
                if control.unit == 'processing_unit':
                    image_processing.append(c)
                else:
                    sensor_control.append(c)

        ui_elements.append(sensor_control)
        if image_processing.elements:
            ui_elements.append(image_processing)
        ui_elements.append(ui.Button("refresh", gui_update_from_device))
        ui_elements.append(ui.Button("load defaults", gui_load_defaults))
        self.g_pool.capture_source_menu.extend(ui_elements)


    def cleanup(self):
        self.devices.cleanup()
        self.devices = None
        if self.uvc_capture_left:
            self.uvc_capture_left.close()
            self.uvc_capture_left = None
        if self.uvc_capture_right:
            self.uvc_capture_right.close()
            self.uvc_capture_right = None
        super().cleanup()


class UVCStereoManager(Base_Manager):
    """Manages local USB sources

    Attributes:
        check_intervall (float): Intervall in which to look for new UVC devices
    """

    gui_name = 'Local USB Stereo'

    def __init__(self, g_pool):
        super().__init__(g_pool)
        print("__INIT__ of uvc stereo manager called")
        self.devices = uvc.Device_List()
        self.selectorLeft = {}
        self.selectorRight = {}
        self.selected_source_left = 0
        self.selected_source_right = 0

    def get_init_dict(self):
        return {}

    def setLeftUID(self, source_uid):
        if not source_uid:
            return
        self.selected_source_left = source_uid
        if not uvc.is_accessible(source_uid):
            logger.error("The selected camera is already in use or blocked.")
            return

    def setRightUID(self, source_uid):
        if not source_uid:
            return
        self.selected_source_right = source_uid
        if not uvc.is_accessible(source_uid):
            logger.error("The selected camera is already in use or blocked.")
            return

    def init_gui(self):
        from pyglui import ui
        ui_elements = []
        ui_elements.append(ui.Info_Text('Local UVC sources'))

        def dev_selection_list():
            default = (None, 'Select to activate')
            self.devices.update()
            dev_pairs = [default] + [(d['uid'], d['name'] + '(' + d['uid'] + ')') for d in self.devices]
            logger.debug("devpairs: {}".format(dev_pairs))
            logger.debug("zipped: {}".format(zip(*dev_pairs)))

            return zip(*dev_pairs)

        def activate():
            settings = {
                'frame_size': self.g_pool.capture.frame_size,
                'frame_rate': self.g_pool.capture.frame_rate,
                'uid_left': self.selected_source_left,
                'uid_right': self.selected_source_right,
            }
            print(settings)
            if self.g_pool.process == 'world':
                self.notify_all({'subject': 'start_plugin', "name": "UVCSourceStereo", 'args': settings})
            else:
                self.notify_all(
                    {'subject': 'start_eye_capture', 'target': self.g_pool.process, "name": "UVCSourceStereo",
                     'args': settings})

        self.selectorLeft = ui.Selector(
            'selected_source_left',
            selection_getter=dev_selection_list,
            getter=lambda: None,
            setter=self.setLeftUID,
            label='World  left'
        )
        self.selectorRight = ui.Selector(
            'selected_source_right',
            selection_getter=dev_selection_list,
            getter=lambda: None,
            setter=self.setRightUID,
            label='World Right'
        )
        ui_elements.append(self.selectorLeft)
        ui_elements.append(self.selectorRight)

        ui_elements.append(ui.Button('Activate', activate))

        self.g_pool.capture_selector_menu.extend(ui_elements)

    def cleanup(self):
        self.deinit_gui()
        self.devices.cleanup()
        self.devices = None

    def recent_events(self, events):
        pass
