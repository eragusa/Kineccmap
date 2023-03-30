#!/bin/bash

ps -ef|grep LokyProcess|awk '{print $2}'|xargs kill
