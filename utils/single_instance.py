# Copyright 2024 Akeeal Mohammed

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import fcntl

class SingleInstance:
    def __init__(self):
        self.lockfile = os.path.normpath(os.path.join(os.path.expanduser("~"), f'.{os.path.basename(sys.argv[0])}.lock'))
        self.lock_file_pointer = None

    def try_lock(self):
        try:
            self.lock_file_pointer = open(self.lockfile, 'w')
            fcntl.lockf(self.lock_file_pointer, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except IOError:
            return False

    def unlock(self):
        if self.lock_file_pointer is not None:
            fcntl.lockf(self.lock_file_pointer, fcntl.LOCK_UN)
            self.lock_file_pointer.close()
            os.unlink(self.lockfile)