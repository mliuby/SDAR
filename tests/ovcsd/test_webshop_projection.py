from agent_system.environments.env_package.webshop.projection import webshop_projection


def test_webshop_projection_accepts_action_without_think_when_disabled():
    actions, valids = webshop_projection(["<action>search[red shoes]</action>"], require_think=False)
    assert actions == ["search[red shoes]"]
    assert valids == [1]


def test_webshop_projection_keeps_strict_default_for_existing_trainers():
    _, valids = webshop_projection(["<action>search[red shoes]</action>"])
    assert valids == [0]

    actions, valids = webshop_projection(["<think>find the item</think><action>search[red shoes]</action>"])
    assert actions == ["search[red shoes]"]
    assert valids == [1]
