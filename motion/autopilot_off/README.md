# AUTOPILOT OFF: JMIN "Tesla" 벌스 3D 키네틱 타이포

Blender 4.5 LTS (EEVEE) 파이썬 스크립트로 씬, 카메라, 컴포지팅까지 전부 생성해.
모든 타이밍은 **박자(beat) 단위**로 짜여 있어서, 실제 음원이 들어오면 BPM과 오프셋만 바꾸면 전체가 다시 맞춰져.

> 음원은 개인 소장용이라 레포에 올리지 않아. `renders/`, `*.wav`, `*.mp4`는 gitignore 처리돼 있어.

## 파일

| 파일 | 역할 |
|---|---|
| `build.py` | 엔트리. 씬 빌드, 카메라 베이크(셰이크/펌프), 월드·노출 플래시, 컴포지터, 렌더 |
| `shots.py` | 첫 줄 샷. 지오메트리, 단어별 애니메이션, 카메라 스플라인 |
| `lib.py` | 타이밍(`F(beat)`), 이징, 머티리얼, 3D 텍스트, 키프레임 헬퍼 |
| `guide_beat.py` | 음원이 없을 때 쓰는 가이드 트랩 비트 (같은 그리드) |
| `fonts/` | Archivo Black, Rubik Mono One, Space Mono (전부 SIL OFL) |

## 실행

```bash
# .blend로 저장만 (블렌더에서 열어서 손으로 다듬을 때)
blender -b --factory-startup -P build.py -- --save autopilot.blend

# 전체 렌더 (PNG 시퀀스 → renders/)
blender -b --factory-startup -P build.py -- --render --res 1920x1080 --samples 32

# 특정 박자 스틸만
blender -b --factory-startup -P build.py -- --still 3,38.5,92.5 --res 960x540

# 실제 음원 타이밍 반영 (BPM, 첫 다운비트까지 걸리는 초)
blender -b --factory-startup -P build.py -- --render --bpm 142.5 --offset 0.38

# mp4로 합치기 (+ 음원)
ffmpeg -framerate 24 -i renders/f_%04d.png -i tesla.m4a -c:v libx264 -crf 16 -pix_fmt yuv420p -shortest out.mp4
```

## 지금 범위: 첫 줄만

"Tesla I just pulled off in that fucking Tesla" 한 줄을 도로 위 원테이크 하나로 만든 버전이야. 단어마다 박자가 `shots.py`의 `WORDS`에 있어.

| 박자 | 단어 | 오브젝트 / 카메라 |
|---|---|---|
| 0–2 | (프리롤) | 어두운 도로, 헤드라이트 깜빡이며 켜짐 |
| 2 | TESLA | 거대 글자가 아스팔트를 뚫고 솟음, 웜즈아이 |
| 3 / 3.5 | I / JUST | 도로 페인트 글자 점등, 카메라 크레인업 |
| 4 | PULLED | 페인트 글자, 카메라 발진 |
| 4.75 | OFF | 램프처럼 일어섰다가 카메라가 넘으면 다시 쾅 |
| 6 / 6.5 | IN / THAT | 갠트리 표지판 점등 |
| 7 | FUCKING | 레드 네온 스트로브, 갠트리 아래 통과 |
| 8 | TESLA | 하늘에서 두 번째 TESLA 낙하 |
| 9.4–10.5 | (꼬리) | E/S 사이 관통, 화이트아웃 |

## 음원 들어오면

1. 실제 BPM과 첫 박 오프셋을 재서 `--bpm` / `--offset`으로 넘기기
2. 가사 타이밍이 그리드랑 다르면 `shots.py`의 `WORDS`에서 단어별 박자를 조정
3. `build.py`의 기본 킥/스네어 그리드(`Global groove`)를 실제 드럼 온셋으로 교체
