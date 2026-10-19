# AUTOPILOT OFF: JMIN "Tesla" 벌스 3D 키네틱 타이포

Blender 4.5 LTS (EEVEE) 파이썬 스크립트 하나로 씬 15개, 카메라, HUD, 컴포지팅까지 전부 생성해.
모든 타이밍은 **박자(beat) 단위**로 짜여 있어서, 실제 음원이 들어오면 BPM과 오프셋만 바꾸면 전체가 다시 맞춰져.

> 음원은 개인 소장용이라 레포에 올리지 않아. `renders/`, `*.wav`, `*.mp4`는 gitignore 처리돼 있어.

## 파일

| 파일 | 역할 |
|---|---|
| `build.py` | 엔트리. 씬 빌드, 카메라 베이크(셰이크/펌프), 월드·노출 플래시, HUD, 컴포지터, 렌더 |
| `shots.py` | 샷 15개. 각 샷이 지오메트리, 애니메이션, `cam(t)`를 직접 정의함 |
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

HUD 텍스트(가사, 배터리, 속도)는 렌더 중에 프레임 핸들러로 갱신돼. 그래서 저장된 .blend를 블렌더 UI에서 그냥 렌더하면 HUD가 고정돼 있어. 최종 렌더는 위 명령으로 돌려.

## 샷 구성 (140 BPM 가정, 100박 ≈ 43초)

| 박자 | 샷 | 카메라 |
|---|---|---|
| 0–8 | TESLA | 웜즈아이 → 풀백+크레인업 → E/S 사이 관통 |
| 8–12 | TWO BAD HOES | 미러 월 사이 바렐롤 → 4필러 슬라롬 |
| 12–16 | POP | 푸시인 → 풍선 글자 폭발 슬로모 관통 |
| 16–24 | TALK IT OUT | 말풍선 오빗 → 레드 변색 → 창 팔로우샷 → 디플레이트 |
| 24–32 | POLKA DOT | 9 → 81 → 2197개 인피니티 룸 표류 |
| 32–40 | SLIME MY B ×4 / SLATT | 정면 → 45° → 탑뷰 → 웜즈아이 (녹는 셰이프키) → 수직 분출 |
| 40–48 | TEAM / CREW / SWAG / WACK | 트래킹 + REJECTED 도장 → 180° 휩 → 도미노 → WACK 낙하 |
| 48–52 | EXOTIC | 볼류메트릭 연기 속 디졸브 |
| 52–56 | FLAWLESS | 스펙트럼 링 오빗 → 다이브 |
| 56–64 | MILEAGE → MILEY | 오도미터 매크로 → 슬롯 착지 → 레킹볼 폭파 |
| 64–68 | GEEKED | 1, 2 팝 → 돌리 줌(버티고) + 더치 + 색수차 |
| 68–72 | SHE SEE | 조리개 통과 + 랙 포커스 |
| 72–80 | HELL NAW / HE'LL NAW | 차단기 급정거 ×2 → 아포스트로피 낙하 사이드아이 |
| 80–88 | OLD HEADS / DROWN | 파도 로우 트래킹 → 수중 다이브 |
| 88–100 | NEW GENERATION / YOUNG SENSATION | 수면 돌파 상승 → 단어 수렴 → 화이트 플래시 → 오빗, AUTOPILOT: OFF |

## 음원 들어오면

1. 실제 BPM과 첫 박 오프셋을 재서 `--bpm` / `--offset`으로 넘기기
2. 가사 타이밍이 그리드랑 다르면 `shots.py`에서 각 샷의 `length`와 `hit()` / `lyric()` 박자를 조정
3. `build.py`의 기본 킥/스네어 그리드(`Global groove`)를 실제 드럼 온셋으로 교체
