#!/bin/bash
# Container entrypoint: install the public key passed in $PUBLIC_KEY (if any),
# start sshd, and stay up.
set -e
mkdir -p /root/.ssh
if [ -n "$PUBLIC_KEY" ]; then
  echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi
ssh-keygen -A >/dev/null 2>&1 || true
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config || true
/usr/sbin/sshd -D &
exec sleep infinity
